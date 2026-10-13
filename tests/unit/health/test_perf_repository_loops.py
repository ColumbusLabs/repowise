"""Unit tests for the typed Java repository / mapper call in a loop.

Each fixture is a small Java tree parsed by the real ingestion parser, so the
heritage, imports, annotations and type references the pass reads are the ones
an index would hold. ``walked`` lists only the file being scored.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from repowise.core.analysis.health.complexity import FileComplexity, PerfHit
from repowise.core.analysis.health.perf.lazy_load import collect_lazy_loads
from repowise.core.analysis.health.perf.repository_loops import persistence_types
from repowise.core.ingestion.models import FileInfo
from repowise.core.ingestion.parser import parse_file


def _require_java() -> None:
    from repowise.core.ingestion.parser import _get_language

    if _get_language("java") is None:
        pytest.skip("tree-sitter language pack missing for java")


def _parse(tmp_path: Path, rel: str, src: str, *, is_test: bool = False):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    info = FileInfo(
        path=rel, abs_path=str(p), language="java", size_bytes=p.stat().st_size, git_hash="",
        last_modified=datetime.now(), is_test=is_test, is_config=False, is_api_contract=False,
        is_entry_point=False,
    )
    return parse_file(info, p.read_bytes())


def _hits(tmp_path: Path, files: dict[str, str], scored: str, *, is_test: bool = False,
          seeded: list[PerfHit] | None = None) -> list[PerfHit]:
    _require_java()
    parsed = {rel: _parse(tmp_path, rel, src, is_test=is_test and rel == scored)
              for rel, src in files.items()}
    fcx = FileComplexity(functions=[], classes=[])
    fcx.perf_hits.extend(seeded or [])
    collect_lazy_loads([(parsed[scored], fcx)], list(parsed.values()))
    return [h for h in fcx.perf_hits if h.kind == "io_in_loop" and h not in (seeded or [])]


_SCAN_CONFIG = """
    package com.shop.config;
    import org.mybatis.spring.annotation.MapperScan;
    @MapperScan({"com.shop.mapper"})
    public class MyBatisConfig {}
"""
_BRAND_MAPPER = """
    package com.shop.mapper;
    public interface BrandMapper { int insert(Brand row); }
"""
_SERVICE = "src/com/shop/service/BrandService.java"


def _mybatis(service_src: str) -> dict[str, str]:
    return {
        "src/com/shop/config/MyBatisConfig.java": _SCAN_CONFIG,
        "src/com/shop/mapper/BrandMapper.java": _BRAND_MAPPER,
        _SERVICE: service_src,
    }


def test_mapper_scan_package_insert_per_row(tmp_path: Path):
    hits = _hits(tmp_path, _mybatis("""
        package com.shop.service;
        import com.shop.mapper.BrandMapper;
        public class BrandService {
            private BrandMapper brandMapper;
            public int create(java.util.List<Brand> rows) {
                for (Brand row : rows) {
                    row.setSort(0);
                    brandMapper.insert(row);
                }
                return rows.size();
            }
        }
    """), _SERVICE)
    assert len(hits) == 1
    hit = hits[0]
    assert (hit.line, hit.function, hit.detail, hit.path) == (9, "create", "db", ())
    assert hit.sink_call == "brandMapper.insert"
    assert hit.loop_line == 7


def test_spring_data_repository_through_own_base(tmp_path: Path):
    files = {
        "src/a/BaseRepository.java": """
            package a;
            import org.springframework.data.jpa.repository.JpaRepository;
            public interface BaseRepository<T> extends JpaRepository<T, Long> {}
        """,
        "src/a/ActivityRepository.java": """
            package a;
            public interface ActivityRepository extends BaseRepository<Activity> {}
        """,
        "src/a/Adapter.java": """
            package a;
            public class Adapter {
                private final ActivityRepository activities;
                void update(Account account) {
                    for (Activity a : account.getActivities()) {
                        this.activities.persist(a);
                    }
                }
            }
        """,
    }
    hits = _hits(tmp_path, files, "src/a/Adapter.java")
    assert [(h.line, h.function) for h in hits] == [(7, "update")]


def test_mapper_annotation_and_parameter_receiver(tmp_path: Path):
    files = {
        "src/b/StockMapper.java": """
            package b;
            import org.apache.ibatis.annotations.Mapper;
            @Mapper
            public interface StockMapper { Stock selectByPrimaryKey(Long id); }
        """,
        "src/b/Locker.java": """
            package b;
            public class Locker {
                void lock(StockMapper mapper, java.util.List<Item> items) {
                    for (Item item : items) {
                        Stock s = mapper.selectByPrimaryKey(item.getSkuId());
                    }
                }
            }
        """,
    }
    assert [h.line for h in _hits(tmp_path, files, "src/b/Locker.java")] == [6]


def test_repository_named_interface_without_spring_base_is_not_typed(tmp_path: Path):
    files = {
        "src/c/JpaRepository.java": "package c;\npublic interface JpaRepository<T, K> {}\n",
        "src/c/UserRepository.java": """
            package c;
            public interface UserRepository extends JpaRepository<User, Long> {}
        """,
        "src/c/Svc.java": """
            package c;
            public class Svc {
                private UserRepository userRepository;
                void f(java.util.List<User> users) {
                    for (User u : users) { userRepository.save(u); }
                }
            }
        """,
    }
    assert _hits(tmp_path, files, "src/c/Svc.java") == []


@pytest.mark.parametrize(
    "body",
    [
        # Header call runs once.
        "for (Brand b : brandMapper.selectAll()) { b.setSort(0); }",
        # A local of another type shadows the typed field.
        "java.util.Map<Long, Brand> brandMapper = null;\n"
        "for (Brand b : rows) { brandMapper.get(b.getId()); }",
        # A constant iterable is a fixed number of calls.
        "for (Long id : List.of(1L, 2L)) { brandMapper.insert(null); }",
    ],
)
def test_not_a_per_row_call(tmp_path: Path, body: str):
    service = (
        "package com.shop.service;\nimport com.shop.mapper.BrandMapper;\n"
        "public class BrandService {\n  private BrandMapper brandMapper;\n"
        "  void f(java.util.List<Brand> rows) {\n" + body + "\n  }\n}\n"
    )
    assert _hits(tmp_path, _mybatis(service), _SERVICE) == []


_LOOP_SERVICE = """
    package com.shop.service;
    import com.shop.mapper.BrandMapper;
    public class BrandService {
        private BrandMapper brandMapper;
        void f(java.util.List<Brand> rows) {
            for (Brand b : rows) { brandMapper.insert(b); }
        }
    }
"""


def test_test_files_are_skipped(tmp_path: Path):
    assert _hits(tmp_path, _mybatis(_LOOP_SERVICE), _SERVICE, is_test=True) == []


def test_line_the_walker_flagged_is_not_repeated(tmp_path: Path):
    seeded = [PerfHit("io_in_loop", 7, "f", "db")]
    assert _hits(tmp_path, _mybatis(_LOOP_SERVICE), _SERVICE, seeded=seeded) == []


def test_ambiguous_simple_name_is_dropped(tmp_path: Path):
    _require_java()
    files = _mybatis(_LOOP_SERVICE)
    files["src/com/other/BrandMapper.java"] = "package com.other;\npublic class BrandMapper {}\n"
    parsed = [_parse(tmp_path, rel, src) for rel, src in files.items()]
    assert "BrandMapper" not in persistence_types(parsed)
