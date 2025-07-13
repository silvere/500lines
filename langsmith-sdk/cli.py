"""
LangSmith Agent SDK CLI工具
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

from langsmith_agent_sdk import LangSmithAgentSDK
from rich.console import Console
from rich.table import Table


def main():
    """主命令行入口"""
    parser = argparse.ArgumentParser(
        description="LangSmith Agent SDK CLI工具"
    )
    
    # 全局参数
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="LangSmith API密钥（也可以通过LANGSMITH_API_KEY环境变量设置）"
    )
    parser.add_argument(
        "--project", 
        type=str, 
        default="agent-traces",
        help="项目名称（默认: agent-traces）"
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # analytics子命令
    analytics_parser = subparsers.add_parser("analytics", help="获取trace分析数据")
    analytics_parser.add_argument(
        "--start-date", 
        type=str, 
        help="开始日期 (YYYY-MM-DD格式)"
    )
    analytics_parser.add_argument(
        "--end-date", 
        type=str, 
        help="结束日期 (YYYY-MM-DD格式)"
    )
    analytics_parser.add_argument(
        "--output", 
        type=str, 
        help="输出文件路径"
    )
    
    # export子命令
    export_parser = subparsers.add_parser("export", help="导出trace数据")
    export_parser.add_argument(
        "output_file", 
        type=str, 
        help="输出文件路径"
    )
    export_parser.add_argument(
        "--format", 
        type=str, 
        choices=["json", "csv"], 
        default="json",
        help="导出格式（默认: json）"
    )
    
    # list子命令
    list_parser = subparsers.add_parser("list", help="列出项目信息")
    list_parser.add_argument(
        "--type", 
        type=str, 
        choices=["projects", "datasets", "runs"], 
        default="projects",
        help="列出的类型（默认: projects）"
    )
    list_parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="限制结果数量（默认: 10）"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 初始化SDK
    try:
        sdk = LangSmithAgentSDK(
            api_key=args.api_key,
            project_name=args.project
        )
    except Exception as e:
        console = Console()
        console.print(f"[red]错误: 无法初始化SDK - {str(e)}[/red]")
        sys.exit(1)
    
    # 执行命令
    try:
        if args.command == "analytics":
            handle_analytics(sdk, args)
        elif args.command == "export":
            handle_export(sdk, args)
        elif args.command == "list":
            handle_list(sdk, args)
    except Exception as e:
        console = Console()
        console.print(f"[red]错误: {str(e)}[/red]")
        sys.exit(1)


def handle_analytics(sdk: LangSmithAgentSDK, args):
    """处理analytics命令"""
    console = Console()
    
    # 解析日期
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # 获取分析数据
    analytics = sdk.get_trace_analytics(
        project_name=args.project,
        start_date=start_date,
        end_date=end_date
    )
    
    # 显示结果
    console.print(f"[bold green]项目分析: {args.project}[/bold green]")
    
    # 基本统计
    table = Table(title="基本统计")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("总运行次数", str(analytics["total_runs"]))
    table.add_row("成功运行", str(analytics["successful_runs"]))
    table.add_row("失败运行", str(analytics["failed_runs"]))
    table.add_row("平均执行时间", f"{analytics['avg_duration']:.2f}秒")
    
    if analytics["total_runs"] > 0:
        success_rate = analytics["successful_runs"] / analytics["total_runs"] * 100
        table.add_row("成功率", f"{success_rate:.1f}%")
    
    console.print(table)
    
    # Agent统计
    if analytics["agents"]:
        console.print("\n[bold]Agent统计:[/bold]")
        agent_table = Table()
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("运行次数", style="blue")
        agent_table.add_column("成功次数", style="green")
        agent_table.add_column("失败次数", style="red")
        agent_table.add_column("成功率", style="yellow")
        
        for agent_name, stats in analytics["agents"].items():
            success_rate = stats["success"] / max(stats["count"], 1) * 100
            agent_table.add_row(
                agent_name,
                str(stats["count"]),
                str(stats["success"]),
                str(stats["failed"]),
                f"{success_rate:.1f}%"
            )
        
        console.print(agent_table)
    
    # 错误统计
    if analytics["errors"]:
        console.print("\n[bold]错误统计:[/bold]")
        error_table = Table()
        error_table.add_column("错误类型", style="cyan")
        error_table.add_column("次数", style="red")
        
        for error_type, count in analytics["errors"].items():
            error_table.add_row(error_type, str(count))
        
        console.print(error_table)
    
    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"\n[green]分析结果已保存到: {args.output}[/green]")


def handle_export(sdk: LangSmithAgentSDK, args):
    """处理export命令"""
    console = Console()
    
    console.print(f"[yellow]导出trace数据到: {args.output_file}[/yellow]")
    
    sdk.export_traces(
        output_file=args.output_file,
        project_name=args.project,
        format=args.format
    )
    
    console.print(f"[green]导出完成! 格式: {args.format}[/green]")


def handle_list(sdk: LangSmithAgentSDK, args):
    """处理list命令"""
    console = Console()
    
    if args.type == "projects":
        console.print("[yellow]列出项目功能正在开发中...[/yellow]")
        # TODO: 实现项目列表功能
        
    elif args.type == "datasets":
        console.print("[yellow]列出数据集功能正在开发中...[/yellow]")
        # TODO: 实现数据集列表功能
        
    elif args.type == "runs":
        console.print(f"[yellow]列出项目 {args.project} 的运行记录...[/yellow]")
        
        # 获取运行记录
        runs = list(sdk.client.list_runs(
            project_name=args.project,
            limit=args.limit
        ))
        
        if not runs:
            console.print("[red]没有找到运行记录[/red]")
            return
        
        # 显示运行记录
        table = Table(title=f"最近 {len(runs)} 条运行记录")
        table.add_column("ID", style="cyan")
        table.add_column("Agent", style="blue")
        table.add_column("状态", style="green")
        table.add_column("开始时间", style="yellow")
        table.add_column("持续时间", style="magenta")
        
        for run in runs:
            status = "成功" if run.error is None else "失败"
            status_color = "green" if run.error is None else "red"
            
            duration = ""
            if run.start_time and run.end_time:
                duration = f"{(run.end_time - run.start_time).total_seconds():.2f}s"
            
            table.add_row(
                str(run.id)[:8] + "...",
                run.name or "Unknown",
                f"[{status_color}]{status}[/{status_color}]",
                run.start_time.strftime("%Y-%m-%d %H:%M:%S") if run.start_time else "Unknown",
                duration
            )
        
        console.print(table)


if __name__ == "__main__":
    main()