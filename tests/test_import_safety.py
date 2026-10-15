import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


# 使用绝对路径导入模块，确保测试不依赖启动目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportSafetyTests(unittest.TestCase):
    def assert_import_is_quiet_and_side_effect_free(self, module_name):
        """验证导入过程无输出、无缓存且不要求可选训练依赖"""
        module_path = PROJECT_ROOT / f"{module_name}.py"

        # 在独立子进程中导入，避免当前测试进程的模块缓存干扰结果
        script = textwrap.dedent(
            f"""
            import importlib.util

            spec = importlib.util.spec_from_file_location({module_name!r}, {str(module_path)!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            """
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            cache_path = temporary_path / "cache"

            # 将 Hugging Face 缓存重定向到临时目录，便于检测意外下载
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HOME": str(cache_path),
                    "HF_DATASETS_CACHE": str(cache_path / "datasets"),
                    "TRANSFORMERS_CACHE": str(cache_path / "transformers"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=temporary_path,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            # 导入成功且临时目录为空，说明没有输出、下载或写入缓存
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertEqual(list(temporary_path.iterdir()), [])

    def test_dataset_import_has_no_optional_dependency_or_io_side_effect(self):
        self.assert_import_is_quiet_and_side_effect_free("dataset")

    def test_prepare_data_import_has_no_optional_dependency_or_io_side_effect(self):
        self.assert_import_is_quiet_and_side_effect_free("prepare_data")


if __name__ == "__main__":
    unittest.main()
