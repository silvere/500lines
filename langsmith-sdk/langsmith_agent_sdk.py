"""
LangSmith Agent SDK
一个用于agent trace和evaluation的便捷SDK
"""

import os
import uuid
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import contextlib
from functools import wraps

from langsmith import Client
from langsmith.run_trees import RunTree
from langsmith.schemas import Run, Example
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID


class AgentTrace(BaseModel):
    """Agent执行trace的数据模型"""
    trace_id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None


class EvaluationResult(BaseModel):
    """评估结果的数据模型"""
    evaluation_id: str
    agent_name: str
    dataset_name: str
    scores: Dict[str, float]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    created_at: datetime


class LangSmithAgentSDK:
    """LangSmith Agent SDK主类"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 project_name: str = "agent-traces",
                 endpoint: Optional[str] = None):
        """
        初始化SDK
        
        Args:
            api_key: LangSmith API密钥，如果不提供将从环境变量LANGSMITH_API_KEY读取
            project_name: 项目名称
            endpoint: LangSmith端点，默认使用官方端点
        """
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供LangSmith API密钥")
        
        self.project_name = project_name
        self.endpoint = endpoint
        
        # 初始化LangSmith客户端
        self.client = Client(
            api_key=self.api_key,
            api_url=endpoint
        )
        
        # 创建项目（如果不存在）
        try:
            self.client.create_project(project_name=self.project_name)
        except Exception:
            # 项目可能已存在，忽略错误
            pass
            
        self.console = Console()
        self.active_traces: Dict[str, AgentTrace] = {}
        
    def create_trace(self, 
                    agent_name: str,
                    inputs: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        创建一个新的agent trace
        
        Args:
            agent_name: Agent名称
            inputs: 输入数据
            metadata: 元数据
            
        Returns:
            trace_id: trace的唯一标识符
        """
        trace_id = str(uuid.uuid4())
        
        trace = AgentTrace(
            trace_id=trace_id,
            agent_name=agent_name,
            start_time=datetime.now(),
            inputs=inputs,
            metadata=metadata or {}
        )
        
        self.active_traces[trace_id] = trace
        
        # 创建LangSmith运行
        run = self.client.create_run(
            name=agent_name,
            project_name=self.project_name,
            run_type="chain",
            inputs=inputs,
            extra=metadata or {}
        )
        
        trace.metadata["langsmith_run_id"] = str(run.id)
        
        self.console.print(f"[green]✓[/green] 创建trace: {trace_id} ({agent_name})")
        return trace_id
        
    def add_step(self, 
                trace_id: str,
                step_name: str,
                inputs: Dict[str, Any],
                outputs: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        向trace添加执行步骤
        
        Args:
            trace_id: trace ID
            step_name: 步骤名称
            inputs: 步骤输入
            outputs: 步骤输出
            metadata: 步骤元数据
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} 不存在")
            
        trace = self.active_traces[trace_id]
        
        step = {
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {}
        }
        
        trace.steps.append(step)
        
        # 添加到LangSmith
        if "langsmith_run_id" in trace.metadata:
            parent_run_id = trace.metadata["langsmith_run_id"]
            child_run = self.client.create_run(
                name=step_name,
                project_name=self.project_name,
                run_type="tool",
                inputs=inputs,
                outputs=outputs,
                parent_run_id=parent_run_id,
                extra=metadata or {}
            )
            step["langsmith_run_id"] = str(child_run.id)
            
        self.console.print(f"[blue]→[/blue] 添加步骤: {step_name}")
        
    def finish_trace(self, 
                    trace_id: str,
                    outputs: Dict[str, Any],
                    error: Optional[str] = None,
                    total_tokens: Optional[int] = None,
                    cost: Optional[float] = None):
        """
        完成trace
        
        Args:
            trace_id: trace ID
            outputs: 最终输出
            error: 错误信息（如果有）
            total_tokens: 总token数
            cost: 成本
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} 不存在")
            
        trace = self.active_traces[trace_id]
        trace.end_time = datetime.now()
        trace.outputs = outputs
        trace.error = error
        trace.total_tokens = total_tokens
        trace.cost = cost
        
        # 更新LangSmith运行
        if "langsmith_run_id" in trace.metadata:
            run_id = trace.metadata["langsmith_run_id"]
            self.client.update_run(
                run_id=run_id,
                outputs=outputs,
                error=error,
                end_time=trace.end_time
            )
            
        status = "[red]✗[/red]" if error else "[green]✓[/green]"
        duration = (trace.end_time - trace.start_time).total_seconds()
        
        self.console.print(f"{status} 完成trace: {trace_id} ({duration:.2f}s)")
        
        # 移除活跃trace
        del self.active_traces[trace_id]
        
        return trace
        
    @contextlib.contextmanager
    def trace_context(self, 
                     agent_name: str,
                     inputs: Dict[str, Any],
                     metadata: Optional[Dict[str, Any]] = None):
        """
        上下文管理器，用于自动管理trace生命周期
        
        Args:
            agent_name: Agent名称
            inputs: 输入数据
            metadata: 元数据
            
        Yields:
            trace_id: trace ID
        """
        trace_id = self.create_trace(agent_name, inputs, metadata)
        
        try:
            yield trace_id
        except Exception as e:
            self.finish_trace(trace_id, {}, error=str(e))
            raise
        else:
            # 如果没有手动调用finish_trace，使用默认输出
            if trace_id in self.active_traces:
                self.finish_trace(trace_id, {"status": "completed"})
                
    def trace_decorator(self, 
                       agent_name: Optional[str] = None,
                       extract_inputs: Optional[Callable] = None,
                       extract_outputs: Optional[Callable] = None):
        """
        装饰器，用于自动trace函数执行
        
        Args:
            agent_name: Agent名称，如果不提供则使用函数名
            extract_inputs: 从函数参数提取输入的函数
            extract_outputs: 从函数返回值提取输出的函数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = agent_name or func.__name__
                
                # 提取输入
                if extract_inputs:
                    inputs = extract_inputs(*args, **kwargs)
                else:
                    inputs = {"args": args, "kwargs": kwargs}
                
                with self.trace_context(name, inputs) as trace_id:
                    result = func(*args, **kwargs)
                    
                    # 提取输出
                    if extract_outputs:
                        outputs = extract_outputs(result)
                    else:
                        outputs = {"result": result}
                    
                    self.finish_trace(trace_id, outputs)
                    return result
                    
            return wrapper
        return decorator
        
    def create_dataset(self, 
                      dataset_name: str,
                      examples: List[Dict[str, Any]]) -> str:
        """
        创建评估数据集
        
        Args:
            dataset_name: 数据集名称
            examples: 示例数据列表
            
        Returns:
            dataset_id: 数据集ID
        """
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=f"Agent evaluation dataset: {dataset_name}"
            )
            
            # 添加示例
            for example in examples:
                self.client.create_example(
                    dataset_id=dataset.id,
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                    metadata=example.get("metadata", {})
                )
                
            self.console.print(f"[green]✓[/green] 创建数据集: {dataset_name} ({len(examples)}个示例)")
            return str(dataset.id)
            
        except Exception as e:
            self.console.print(f"[red]✗[/red] 创建数据集失败: {str(e)}")
            raise
            
    async def evaluate_agent(self,
                           agent_func: Callable,
                           dataset_name: str,
                           evaluators: List[Callable],
                           agent_name: Optional[str] = None) -> EvaluationResult:
        """
        评估agent性能
        
        Args:
            agent_func: 要评估的agent函数
            dataset_name: 数据集名称
            evaluators: 评估函数列表
            agent_name: Agent名称
            
        Returns:
            EvaluationResult: 评估结果
        """
        evaluation_id = str(uuid.uuid4())
        agent_name = agent_name or "agent"
        
        self.console.print(f"[yellow]▶[/yellow] 开始评估: {agent_name}")
        
        # 获取数据集
        dataset = self.client.read_dataset(dataset_name=dataset_name)
        examples = list(self.client.list_examples(dataset_id=dataset.id))
        
        results = []
        scores = {}
        
        with Progress() as progress:
            task = progress.add_task(f"评估 {agent_name}", total=len(examples))
            
            for example in examples:
                progress.update(task, advance=1)
                
                # 运行agent
                with self.trace_context(agent_name, example.inputs) as trace_id:
                    try:
                        agent_output = agent_func(example.inputs)
                        self.finish_trace(trace_id, {"output": agent_output})
                        
                        # 运行评估器
                        eval_scores = {}
                        for evaluator in evaluators:
                            score = evaluator(example.inputs, agent_output, example.outputs)
                            eval_scores[evaluator.__name__] = score
                            
                        results.append({
                            "example_id": str(example.id),
                            "inputs": example.inputs,
                            "expected": example.outputs,
                            "actual": agent_output,
                            "scores": eval_scores
                        })
                        
                        # 累计分数
                        for metric, score in eval_scores.items():
                            if metric not in scores:
                                scores[metric] = []
                            scores[metric].append(score)
                            
                    except Exception as e:
                        self.finish_trace(trace_id, {}, error=str(e))
                        results.append({
                            "example_id": str(example.id),
                            "inputs": example.inputs,
                            "expected": example.outputs,
                            "actual": None,
                            "error": str(e),
                            "scores": {}
                        })
                        
        # 计算平均分数
        avg_scores = {}
        for metric, score_list in scores.items():
            if score_list:
                avg_scores[metric] = sum(score_list) / len(score_list)
                
        # 创建评估结果
        evaluation_result = EvaluationResult(
            evaluation_id=evaluation_id,
            agent_name=agent_name,
            dataset_name=dataset_name,
            scores=avg_scores,
            results=results,
            summary={
                "total_examples": len(examples),
                "successful_runs": len([r for r in results if "error" not in r]),
                "failed_runs": len([r for r in results if "error" in r])
            },
            created_at=datetime.now()
        )
        
        # 显示结果
        self._display_evaluation_results(evaluation_result)
        
        return evaluation_result
        
    def _display_evaluation_results(self, result: EvaluationResult):
        """显示评估结果"""
        self.console.print(f"\n[bold green]评估完成: {result.agent_name}[/bold green]")
        
        # 创建结果表格
        table = Table(title="评估结果")
        table.add_column("指标", style="cyan")
        table.add_column("分数", style="green")
        
        for metric, score in result.scores.items():
            table.add_row(metric, f"{score:.4f}")
            
        self.console.print(table)
        
        # 显示摘要
        summary = result.summary
        self.console.print(f"\n[bold]摘要:[/bold]")
        self.console.print(f"总示例数: {summary['total_examples']}")
        self.console.print(f"成功运行: {summary['successful_runs']}")
        self.console.print(f"失败运行: {summary['failed_runs']}")
        
    def get_trace_analytics(self, 
                          project_name: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取trace分析数据
        
        Args:
            project_name: 项目名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            分析数据
        """
        project_name = project_name or self.project_name
        
        # 获取运行数据
        runs = list(self.client.list_runs(
            project_name=project_name,
            start_time=start_date,
            end_time=end_date
        ))
        
        # 分析数据
        analytics = {
            "total_runs": len(runs),
            "successful_runs": len([r for r in runs if r.error is None]),
            "failed_runs": len([r for r in runs if r.error is not None]),
            "avg_duration": 0,
            "agents": {},
            "errors": {}
        }
        
        durations = []
        for run in runs:
            if run.start_time and run.end_time:
                duration = (run.end_time - run.start_time).total_seconds()
                durations.append(duration)
                
            # 统计agent
            agent_name = run.name
            if agent_name not in analytics["agents"]:
                analytics["agents"][agent_name] = {"count": 0, "success": 0, "failed": 0}
            analytics["agents"][agent_name]["count"] += 1
            
            if run.error:
                analytics["agents"][agent_name]["failed"] += 1
                # 统计错误
                error_type = type(run.error).__name__
                if error_type not in analytics["errors"]:
                    analytics["errors"][error_type] = 0
                analytics["errors"][error_type] += 1
            else:
                analytics["agents"][agent_name]["success"] += 1
                
        if durations:
            analytics["avg_duration"] = sum(durations) / len(durations)
            
        return analytics
        
    def export_traces(self, 
                     output_file: str,
                     project_name: Optional[str] = None,
                     format: str = "json") -> None:
        """
        导出trace数据
        
        Args:
            output_file: 输出文件路径
            project_name: 项目名称
            format: 导出格式 (json, csv)
        """
        project_name = project_name or self.project_name
        
        runs = list(self.client.list_runs(project_name=project_name))
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([run.dict() for run in runs], f, indent=2, ensure_ascii=False, default=str)
        elif format == "csv":
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if runs:
                    writer = csv.DictWriter(f, fieldnames=runs[0].dict().keys())
                    writer.writeheader()
                    for run in runs:
                        writer.writerow(run.dict())
                        
        self.console.print(f"[green]✓[/green] 导出完成: {output_file}")


# 常用评估函数
def accuracy_evaluator(inputs: Dict[str, Any], 
                      actual: Any, 
                      expected: Any) -> float:
    """准确率评估器"""
    return 1.0 if actual == expected else 0.0


def similarity_evaluator(inputs: Dict[str, Any], 
                        actual: str, 
                        expected: str) -> float:
    """文本相似度评估器"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, actual, expected).ratio()


def custom_evaluator(evaluation_func: Callable[[Dict[str, Any], Any, Any], float]):
    """自定义评估器装饰器"""
    return evaluation_func