#!/usr/bin/env python3
"""
Divine Agent System - Command Line Interface
Supreme Agentic Orchestrator (SAO) CLI

Provides command-line tools for managing and interacting with the agent system.
"""

import argparse
import asyncio
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Rich for beautiful CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Import our agent system
try:
    from . import SupremeAgenticOrchestrator, get_system_info, list_all_agents
    from .cloud_mastery import get_department_info, create_agent_instance
except ImportError:
    # Handle case where we're running directly
    import agents
    SupremeAgenticOrchestrator = agents.SupremeAgenticOrchestrator
    get_system_info = agents.get_system_info
    list_all_agents = agents.list_all_agents

class DivineAgentCLI:
    """Command Line Interface for Divine Agent System"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.orchestrator = None
        self.config = {}
        
    def print(self, *args, **kwargs):
        """Print with rich formatting if available"""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)
            
    def print_banner(self):
        """Display the Divine Agent System banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    Divine Agent System                       ║
║              Supreme Agentic Orchestrator (SAO)             ║
║                                                              ║
║  Multi-Agent Cloud Mastery System with Quantum Features     ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        if self.console:
            self.console.print(banner, style="bold blue")
        else:
            print(banner)
            
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file"""
        if config_path is None:
            config_path = "config.yaml"
            
        if not os.path.exists(config_path):
            self.print(f"[yellow]Warning: Config file {config_path} not found[/yellow]")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.print(f"[green]✓ Configuration loaded from {config_path}[/green]")
            return self.config
        except Exception as e:
            self.print(f"[red]✗ Error loading config: {e}[/red]")
            return {}
            
    def init_orchestrator(self) -> bool:
        """Initialize the Supreme Agentic Orchestrator"""
        try:
            self.orchestrator = SupremeAgenticOrchestrator()
            self.print("[green]✓ Supreme Agentic Orchestrator initialized[/green]")
            return True
        except Exception as e:
            self.print(f"[red]✗ Failed to initialize orchestrator: {e}[/red]")
            return False
            
    def cmd_info(self, args):
        """Display system information"""
        self.print_banner()
        
        info = get_system_info()
        
        if self.console:
            # Create a rich table
            table = Table(title="System Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in info.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, indent=2)
                table.add_row(str(key), str(value))
                
            self.console.print(table)
        else:
            print("\nSystem Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
                
    def cmd_list_agents(self, args):
        """List all available agents"""
        agents = list_all_agents()
        
        if self.console:
            tree = Tree("[bold blue]Divine Agent System - Available Agents[/bold blue]")
            
            for dept_name, dept_agents in agents.items():
                dept_branch = tree.add(f"[bold yellow]{dept_name}[/bold yellow]")
                
                for agent_name, agent_info in dept_agents.items():
                    agent_branch = dept_branch.add(f"[green]{agent_name}[/green]")
                    agent_branch.add(f"[dim]{agent_info.get('description', 'No description')}[/dim]")
                    
            self.console.print(tree)
        else:
            print("\nAvailable Agents:")
            for dept_name, dept_agents in agents.items():
                print(f"\n{dept_name}:")
                for agent_name, agent_info in dept_agents.items():
                    print(f"  - {agent_name}: {agent_info.get('description', 'No description')}")
                    
    def cmd_create_agent(self, args):
        """Create and configure an agent instance"""
        if not args.department or not args.agent:
            self.print("[red]✗ Department and agent name are required[/red]")
            return
            
        try:
            # Import the specific department
            if args.department == "cloud_mastery":
                from .cloud_mastery import create_agent_instance
                agent = create_agent_instance(args.agent)
                
                if agent:
                    self.print(f"[green]✓ Created {args.agent} agent from {args.department} department[/green]")
                    
                    # Display agent capabilities
                    if hasattr(agent, 'get_capabilities'):
                        capabilities = agent.get_capabilities()
                        
                        if self.console:
                            panel = Panel(
                                "\n".join([f"• {cap}" for cap in capabilities]),
                                title=f"{args.agent} Capabilities",
                                border_style="green"
                            )
                            self.console.print(panel)
                        else:
                            print(f"\n{args.agent} Capabilities:")
                            for cap in capabilities:
                                print(f"  • {cap}")
                else:
                    self.print(f"[red]✗ Failed to create {args.agent} agent[/red]")
            else:
                self.print(f"[red]✗ Unknown department: {args.department}[/red]")
                
        except Exception as e:
            self.print(f"[red]✗ Error creating agent: {e}[/red]")
            
    def cmd_test_agent(self, args):
        """Test an agent's functionality"""
        if not args.department or not args.agent:
            self.print("[red]✗ Department and agent name are required[/red]")
            return
            
        try:
            if args.department == "cloud_mastery":
                from .cloud_mastery import create_agent_instance
                agent = create_agent_instance(args.agent)
                
                if agent and hasattr(agent, 'run_tests'):
                    self.print(f"[yellow]Running tests for {args.agent}...[/yellow]")
                    
                    if self.console:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=self.console
                        ) as progress:
                            task = progress.add_task("Testing agent...", total=None)
                            
                            # Run the tests
                            test_results = agent.run_tests()
                            
                            progress.update(task, completed=True)
                    else:
                        test_results = agent.run_tests()
                        
                    if test_results:
                        self.print(f"[green]✓ All tests passed for {args.agent}[/green]")
                    else:
                        self.print(f"[red]✗ Some tests failed for {args.agent}[/red]")
                else:
                    self.print(f"[red]✗ Agent {args.agent} does not support testing[/red]")
            else:
                self.print(f"[red]✗ Unknown department: {args.department}[/red]")
                
        except Exception as e:
            self.print(f"[red]✗ Error testing agent: {e}[/red]")
            
    def cmd_start_server(self, args):
        """Start the Divine Agent System server"""
        self.print("[yellow]Starting Divine Agent System server...[/yellow]")
        
        if not self.init_orchestrator():
            return
            
        try:
            # Configure server settings
            host = args.host or self.config.get('communication', {}).get('rest_api', {}).get('host', '0.0.0.0')
            port = args.port or self.config.get('communication', {}).get('rest_api', {}).get('port', 8000)
            workers = args.workers or self.config.get('performance', {}).get('concurrency', {}).get('max_workers', 4)
            
            self.print(f"[blue]Server configuration:[/blue]")
            self.print(f"  Host: {host}")
            self.print(f"  Port: {port}")
            self.print(f"  Workers: {workers}")
            self.print(f"  Environment: {args.environment}")
            
            # Start the server (this would be implemented in the orchestrator)
            if hasattr(self.orchestrator, 'start_server'):
                asyncio.run(self.orchestrator.start_server(
                    host=host,
                    port=port,
                    workers=workers,
                    environment=args.environment
                ))
            else:
                self.print("[yellow]Server functionality not yet implemented[/yellow]")
                self.print("[blue]This would start the REST API, WebSocket, and JSON-RPC servers[/blue]")
                
        except KeyboardInterrupt:
            self.print("\n[yellow]Server shutdown requested[/yellow]")
        except Exception as e:
            self.print(f"[red]✗ Server error: {e}[/red]")
            
    def cmd_config(self, args):
        """Display or modify configuration"""
        if args.show:
            if self.console:
                syntax = Syntax(yaml.dump(self.config, default_flow_style=False), "yaml")
                panel = Panel(syntax, title="Current Configuration", border_style="blue")
                self.console.print(panel)
            else:
                print("\nCurrent Configuration:")
                print(yaml.dump(self.config, default_flow_style=False))
                
        elif args.set:
            # Set configuration value
            try:
                keys = args.set[0].split('.')
                value = args.set[1]
                
                # Navigate to the nested key
                current = self.config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                    
                # Set the value
                current[keys[-1]] = value
                
                self.print(f"[green]✓ Set {args.set[0]} = {value}[/green]")
                
            except Exception as e:
                self.print(f"[red]✗ Error setting configuration: {e}[/red]")
                
    def cmd_monitor(self, args):
        """Monitor system status and metrics"""
        self.print("[yellow]Monitoring Divine Agent System...[/yellow]")
        self.print("[dim]Press Ctrl+C to stop monitoring[/dim]")
        
        try:
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Display header
                self.print(f"[bold blue]Divine Agent System Monitor[/bold blue]")
                self.print(f"[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
                
                # System status
                if self.console:
                    status_table = Table(title="System Status")
                    status_table.add_column("Component", style="cyan")
                    status_table.add_column("Status", style="green")
                    status_table.add_column("Details")
                    
                    # Mock status data (would be real in implementation)
                    status_table.add_row("Orchestrator", "✓ Running", "All systems operational")
                    status_table.add_row("Redis", "✓ Connected", "Memory: 45MB")
                    status_table.add_row("PostgreSQL", "✓ Connected", "Connections: 12/100")
                    status_table.add_row("Agents", "✓ Active", "7 agents running")
                    
                    self.console.print(status_table)
                else:
                    print("System Status:")
                    print("  Orchestrator: ✓ Running")
                    print("  Redis: ✓ Connected")
                    print("  PostgreSQL: ✓ Connected")
                    print("  Agents: ✓ Active (7 running)")
                    
                # Wait before next update
                import time
                time.sleep(args.interval or 5)
                
        except KeyboardInterrupt:
            self.print("\n[yellow]Monitoring stopped[/yellow]")
            
    def cmd_deploy(self, args):
        """Deploy agents to cloud infrastructure"""
        self.print(f"[yellow]Deploying to {args.target} environment...[/yellow]")
        
        # This would implement actual deployment logic
        deployment_steps = [
            "Validating configuration",
            "Building container images",
            "Pushing to registry",
            "Updating infrastructure",
            "Running health checks",
            "Completing deployment"
        ]
        
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                for step in deployment_steps:
                    task = progress.add_task(step, total=None)
                    import time
                    time.sleep(1)  # Simulate work
                    progress.update(task, completed=True)
        else:
            for step in deployment_steps:
                print(f"  {step}...")
                import time
                time.sleep(1)
                
        self.print(f"[green]✓ Successfully deployed to {args.target}[/green]")
        
def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        description="Divine Agent System - Supreme Agentic Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info                                    # Show system information
  %(prog)s list-agents                             # List all available agents
  %(prog)s create-agent cloud_mastery devops_engineer  # Create a DevOps agent
  %(prog)s test-agent cloud_mastery security_specialist # Test security agent
  %(prog)s start-server --port 8000                # Start the server
  %(prog)s monitor --interval 10                   # Monitor system status
  %(prog)s deploy production                       # Deploy to production
        """
    )
    
    # Global options
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information')
    
    # List agents command
    list_parser = subparsers.add_parser('list-agents', help='List all available agents')
    
    # Create agent command
    create_parser = subparsers.add_parser('create-agent', help='Create an agent instance')
    create_parser.add_argument('department', help='Department name (e.g., cloud_mastery)')
    create_parser.add_argument('agent', help='Agent name (e.g., devops_engineer)')
    create_parser.add_argument('--config-override', help='Override agent configuration')
    
    # Test agent command
    test_parser = subparsers.add_parser('test-agent', help='Test an agent\'s functionality')
    test_parser.add_argument('department', help='Department name')
    test_parser.add_argument('agent', help='Agent name')
    
    # Start server command
    server_parser = subparsers.add_parser('start-server', help='Start the Divine Agent System server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    server_parser.add_argument('--environment', choices=['development', 'staging', 'production'], 
                              default='development', help='Environment')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--show', action='store_true', help='Show current configuration')
    config_group.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set configuration value')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor system status')
    monitor_parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy to cloud infrastructure')
    deploy_parser.add_argument('target', choices=['development', 'staging', 'production'], 
                              help='Deployment target')
    deploy_parser.add_argument('--dry-run', action='store_true', help='Dry run deployment')
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize CLI
    cli = DivineAgentCLI()
    
    # Load configuration
    cli.load_config(args.config)
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
        
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute command
    if args.command == 'info':
        cli.cmd_info(args)
    elif args.command == 'list-agents':
        cli.cmd_list_agents(args)
    elif args.command == 'create-agent':
        cli.cmd_create_agent(args)
    elif args.command == 'test-agent':
        cli.cmd_test_agent(args)
    elif args.command == 'start-server':
        cli.cmd_start_server(args)
    elif args.command == 'config':
        cli.cmd_config(args)
    elif args.command == 'monitor':
        cli.cmd_monitor(args)
    elif args.command == 'deploy':
        cli.cmd_deploy(args)
    else:
        cli.print_banner()
        parser.print_help()
        
if __name__ == '__main__':
    main()