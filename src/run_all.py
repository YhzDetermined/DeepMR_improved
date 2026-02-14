#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量执行DeepMPrior测试脚本 - Python版本
作者：高级开发工程师
使用：在Anaconda Prompt中执行 python batch_processor.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import traceback


class BatchProcessor:
    def __init__(self):
        # 配置路径（使用绝对路径）
        self.work_dir = Path(r"D:\DeepMPrior\src")
        self.dataset_root = Path(r"D:\DeepMPrior\Dataset\all-bugs")
        self.log_file = self.work_dir / "batch_execution_python.log"
        self.error_log = self.work_dir / "error_log_python.txt"

        # 统计信息
        self.total = 0
        self.success = 0
        self.failed = 0

        # 初始化日志文件
        self.init_logs()

    def init_logs(self):
        """初始化日志文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"批量执行开始时间: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

        with open(self.error_log, 'w', encoding='utf-8') as f:
            f.write(f"错误日志 - 开始时间: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

    def log_message(self, message, log_type="info"):
        """记录日志信息

        Args:
            message: 日志消息
            log_type: 日志类型 (info, error, warning)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{log_type.upper()}] {message}"

        # 写入主日志
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")

        # 如果是错误，同时写入错误日志
        if log_type == "error":
            with open(self.error_log, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")

        # 控制台输出
        if log_type == "error":
            print(f"\033[91m{log_entry}\033[0m")  # 红色
        elif log_type == "warning":
            print(f"\033[93m{log_entry}\033[0m")  # 黄色
        else:
            print(log_entry)

    def check_required_files(self, bug_dir):
        """检查必要的文件是否存在

        Args:
            bug_dir: bug目录路径

        Returns:
            tuple: (是否所有文件都存在, 缺失的文件列表)
        """
        required_files = [
            bug_dir / "trained_model.h5",
            bug_dir / "my_test_X_array.npy",
            bug_dir / "my_test_Y_array.npy"
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        return len(missing_files) == 0, missing_files

    def execute_command(self, bug_id, bug_dir):
        """执行单个bug的测试命令

        Args:
            bug_id: bug编号
            bug_dir: bug目录路径

        Returns:
            bool: 执行是否成功
        """
        self.total += 1

        # 记录开始处理
        self.log_message(f"开始处理第 {self.total} 个bug: {bug_id}")
        print(f"{'=' * 60}")
        print(f"[进度 {self.total}] 正在处理bug: {bug_id}")

        # 检查必要文件
        files_ok, missing_files = self.check_required_files(bug_dir)
        if not files_ok:
            self.log_message(f"bug {bug_id} 缺少必要文件: {', '.join(missing_files)}", "warning")
            self.failed += 1
            return False

        # 构建命令
        model_path = bug_dir / "trained_model.h5"
        test_x_path = bug_dir / "my_test_X_array.npy"
        test_y_path = bug_dir / "my_test_Y_array.npy"

        command = [
            "python", "main2.py",
            str(model_path),
            "0.3",
            str(test_x_path),
            str(test_y_path),
            "class",
            "0.001"
        ]

        # 显示执行的命令
        cmd_str = " ".join(command)
        print(f"执行命令: {cmd_str}")
        self.log_message(f"执行命令: {cmd_str}")

        try:
            # 记录开始时间
            start_time = time.time()

            # 执行命令（阻塞执行，等待完成）
            # 注意：subprocess.run会等待命令执行完成
            result = subprocess.run(
                command,
                cwd=self.work_dir,  # 设置工作目录
                capture_output=True,  # 捕获输出
                text=True,  # 以文本形式返回
                encoding='utf-8',
                errors='ignore'
            )

            # 记录执行时间
            elapsed_time = time.time() - start_time

            # 检查执行结果
            if result.returncode == 0:
                self.log_message(f"bug {bug_id} 处理成功 (耗时: {elapsed_time:.2f}秒)")
                self.success += 1

                # 可选：记录标准输出到日志
                if result.stdout:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[输出] {result.stdout[:500]}...\n")  # 只记录前500字符

                return True
            else:
                self.log_message(f"bug {bug_id} 处理失败 (返回码: {result.returncode}, 耗时: {elapsed_time:.2f}秒)",
                                 "error")
                self.log_message(f"错误输出: {result.stderr[:500]}", "error")
                self.failed += 1

                # 详细错误信息
                error_details = f"""
                Bug ID: {bug_id}
                命令: {cmd_str}
                返回码: {result.returncode}
                标准错误: {result.stderr[:1000]}
                """
                with open(self.error_log, 'a', encoding='utf-8') as f:
                    f.write(error_details + "\n" + "=" * 60 + "\n")

                return False

        except Exception as e:
            # 捕获其他异常
            self.log_message(f"执行命令时发生异常: {str(e)}", "error")
            self.log_message(traceback.format_exc(), "error")
            self.failed += 1
            return False

    def run(self):
        """主执行函数"""
        print("=" * 60)
        print("DeepMPrior批量测试脚本 - Python版本")
        print("=" * 60)
        print(f"工作目录: {self.work_dir}")
        print(f"数据集目录: {self.dataset_root}")
        print(f"日志文件: {self.log_file}")
        print(f"错误日志: {self.error_log}")
        print("=" * 60)

        # 检查工作目录是否存在
        if not self.work_dir.exists():
            self.log_message(f"工作目录不存在: {self.work_dir}", "error")
            return

        # 检查数据集目录是否存在
        if not self.dataset_root.exists():
            self.log_message(f"数据集目录不存在: {self.dataset_root}", "error")
            return

        # 获取所有bug目录（只处理一级子目录）
        bug_dirs = []
        for item in self.dataset_root.iterdir():
            if item.is_dir():
                bug_dirs.append(item)

        if not bug_dirs:
            self.log_message("在数据集目录中未找到任何bug目录", "warning")
            return

        print(f"找到 {len(bug_dirs)} 个bug目录")
        print("=" * 60)

        # 按目录名排序（可选）
        bug_dirs.sort()

        # 依次处理每个bug目录
        for bug_dir in bug_dirs:
            bug_id = bug_dir.name
            self.execute_command(bug_id, bug_dir)

            # 可选：添加延迟，确保资源释放
            time.sleep(1)

        # 输出统计信息
        print("=" * 60)
        print("批量执行完成!")
        print(f"总计处理: {self.total} 个bug")
        print(f"成功: {self.success} 个")
        print(f"失败: {self.failed} 个")
        print("=" * 60)

        # 记录到日志
        self.log_message(f"批量执行完成 - 总计: {self.total}, 成功: {self.success}, 失败: {self.failed}")

        # 如果有失败，提示查看错误日志
        if self.failed > 0:
            print(f"\n注意：有 {self.failed} 个bug处理失败，请查看错误日志: {self.error_log}")


if __name__ == "__main__":
    # 创建处理器并运行
    processor = BatchProcessor()

    try:
        processor.run()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        processor.log_message("用户中断执行", "warning")
    except Exception as e:
        print(f"\n发生未预期错误: {str(e)}")
        processor.log_message(f"未预期错误: {str(e)}", "error")
        processor.log_message(traceback.format_exc(), "error")

    # 等待用户按键退出（可选）
    # if sys.platform == "win32":
    #     print("\n按任意键退出...")
    #     import msvcrt
    #
    #     msvcrt.getch()