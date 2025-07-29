#!/usr/bin/env python3
"""
Divine Agent System - Deployment Script
Supreme Agentic Orchestrator (SAO) Deployment Automation

This script automates the deployment of the Divine Agent System to various
cloud platforms and environments with quantum-enhanced orchestration.
"""

import os
import sys
import json
import yaml
import subprocess
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class DivineDeploymentOrchestrator:
    """Supreme deployment orchestrator for the Divine Agent System"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.deployment_id = f"divine-{int(time.time())}"
        self.log_file = f"deployment-{self.deployment_id}.log"
        
    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.log("WARNING: Config file not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration"""
        return {
            'system': {
                'name': 'Divine Agent System',
                'version': '1.0.0',
                'environment': 'production'
            },
            'deployment': {
                'platform': 'docker',
                'replicas': 3,
                'resources': {
                    'cpu': '1000m',
                    'memory': '2Gi'
                }
            }
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log deployment messages"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
    
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute shell command with logging"""
        self.log(f"Executing: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                self.log(f"STDOUT: {result.stdout.strip()}")
            if result.stderr:
                self.log(f"STDERR: {result.stderr.strip()}", "WARNING")
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites"""
        self.log("Checking deployment prerequisites...")
        
        prerequisites = {
            'docker': 'docker --version',
            'docker-compose': 'docker-compose --version',
            'python': 'python3 --version',
            'git': 'git --version'
        }
        
        missing = []
        for tool, command in prerequisites.items():
            try:
                self.run_command(command, check=True)
                self.log(f"✓ {tool} is available")
            except subprocess.CalledProcessError:
                self.log(f"✗ {tool} is not available", "ERROR")
                missing.append(tool)
        
        if missing:
            self.log(f"Missing prerequisites: {', '.join(missing)}", "ERROR")
            return False
        
        self.log("All prerequisites satisfied")
        return True
    
    def build_docker_images(self) -> bool:
        """Build Docker images for the system"""
        self.log("Building Docker images...")
        
        try:
            # Build main application image
            self.run_command("docker build -t divine-agent-system:latest .")
            self.log("✓ Main application image built")
            
            # Build quantum-enhanced image
            self.run_command(
                "docker build -t divine-agent-system:quantum "
                "--target quantum-enhanced ."
            )
            self.log("✓ Quantum-enhanced image built")
            
            # Build multi-cloud image
            self.run_command(
                "docker build -t divine-agent-system:multi-cloud "
                "--target multi-cloud ."
            )
            self.log("✓ Multi-cloud image built")
            
            return True
        except subprocess.CalledProcessError:
            self.log("Failed to build Docker images", "ERROR")
            return False
    
    def deploy_local_docker(self) -> bool:
        """Deploy to local Docker environment"""
        self.log("Deploying to local Docker environment...")
        
        try:
            # Start services with docker-compose
            self.run_command("docker-compose up -d")
            self.log("✓ Services started with docker-compose")
            
            # Wait for services to be ready
            self.log("Waiting for services to be ready...")
            time.sleep(30)
            
            # Check service health
            result = self.run_command("docker-compose ps", check=False)
            if "Up" in result.stdout:
                self.log("✓ Services are running")
                return True
            else:
                self.log("Services failed to start properly", "ERROR")
                return False
                
        except subprocess.CalledProcessError:
            self.log("Failed to deploy to local Docker", "ERROR")
            return False
    
    def deploy_kubernetes(self, namespace: str = "divine-agents") -> bool:
        """Deploy to Kubernetes cluster"""
        self.log(f"Deploying to Kubernetes namespace: {namespace}...")
        
        try:
            # Create namespace
            self.run_command(f"kubectl create namespace {namespace} --dry-run=client -o yaml | kubectl apply -f -")
            self.log(f"✓ Namespace {namespace} ready")
            
            # Generate Kubernetes manifests
            self.generate_k8s_manifests(namespace)
            
            # Apply manifests
            self.run_command(f"kubectl apply -f k8s/ -n {namespace}")
            self.log("✓ Kubernetes manifests applied")
            
            # Wait for deployment
            self.run_command(f"kubectl rollout status deployment/divine-agent-system -n {namespace}")
            self.log("✓ Deployment rolled out successfully")
            
            # Get service information
            result = self.run_command(f"kubectl get services -n {namespace}", check=False)
            self.log(f"Services:\n{result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError:
            self.log("Failed to deploy to Kubernetes", "ERROR")
            return False
    
    def generate_k8s_manifests(self, namespace: str):
        """Generate Kubernetes deployment manifests"""
        self.log("Generating Kubernetes manifests...")
        
        # Create k8s directory
        os.makedirs("k8s", exist_ok=True)
        
        # Deployment manifest
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'divine-agent-system',
                'namespace': namespace,
                'labels': {
                    'app': 'divine-agent-system',
                    'version': 'v1.0.0',
                    'tier': 'application'
                }
            },
            'spec': {
                'replicas': self.config.get('deployment', {}).get('replicas', 3),
                'selector': {
                    'matchLabels': {
                        'app': 'divine-agent-system'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'divine-agent-system'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'divine-agent-system',
                            'image': 'divine-agent-system:latest',
                            'ports': [{
                                'containerPort': 8000,
                                'name': 'http'
                            }],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': 'production'},
                                {'name': 'LOG_LEVEL', 'value': 'INFO'},
                                {'name': 'QUANTUM_ENABLED', 'value': 'true'},
                                {'name': 'CONSCIOUSNESS_LEVEL', 'value': 'divine'}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': '500m',
                                    'memory': '1Gi'
                                },
                                'limits': {
                                    'cpu': '1000m',
                                    'memory': '2Gi'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'divine-agent-system-service',
                'namespace': namespace,
                'labels': {
                    'app': 'divine-agent-system'
                }
            },
            'spec': {
                'selector': {
                    'app': 'divine-agent-system'
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8000,
                    'name': 'http'
                }],
                'type': 'LoadBalancer'
            }
        }
        
        # ConfigMap for configuration
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'divine-agent-config',
                'namespace': namespace
            },
            'data': {
                'config.yaml': yaml.dump(self.config)
            }
        }
        
        # Write manifests
        with open('k8s/deployment.yaml', 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        with open('k8s/service.yaml', 'w') as f:
            yaml.dump(service, f, default_flow_style=False)
        
        with open('k8s/configmap.yaml', 'w') as f:
            yaml.dump(configmap, f, default_flow_style=False)
        
        self.log("✓ Kubernetes manifests generated")
    
    def deploy_aws_ecs(self) -> bool:
        """Deploy to AWS ECS"""
        self.log("Deploying to AWS ECS...")
        
        try:
            # Check AWS CLI
            self.run_command("aws --version")
            
            # Create ECS cluster
            cluster_name = f"divine-agents-{self.deployment_id}"
            self.run_command(f"aws ecs create-cluster --cluster-name {cluster_name}")
            self.log(f"✓ ECS cluster {cluster_name} created")
            
            # Register task definition
            task_definition = self.generate_ecs_task_definition()
            with open('ecs-task-definition.json', 'w') as f:
                json.dump(task_definition, f, indent=2)
            
            self.run_command("aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json")
            self.log("✓ ECS task definition registered")
            
            # Create service
            self.run_command(
                f"aws ecs create-service "
                f"--cluster {cluster_name} "
                f"--service-name divine-agent-service "
                f"--task-definition divine-agent-system:1 "
                f"--desired-count 2"
            )
            self.log("✓ ECS service created")
            
            return True
            
        except subprocess.CalledProcessError:
            self.log("Failed to deploy to AWS ECS", "ERROR")
            return False
    
    def generate_ecs_task_definition(self) -> Dict[str, Any]:
        """Generate ECS task definition"""
        return {
            'family': 'divine-agent-system',
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '1024',
            'memory': '2048',
            'executionRoleArn': 'arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole',
            'containerDefinitions': [{
                'name': 'divine-agent-system',
                'image': 'divine-agent-system:latest',
                'portMappings': [{
                    'containerPort': 8000,
                    'protocol': 'tcp'
                }],
                'environment': [
                    {'name': 'ENVIRONMENT', 'value': 'production'},
                    {'name': 'QUANTUM_ENABLED', 'value': 'true'},
                    {'name': 'CONSCIOUSNESS_LEVEL', 'value': 'divine'}
                ],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': '/ecs/divine-agent-system',
                        'awslogs-region': 'us-east-1',
                        'awslogs-stream-prefix': 'ecs'
                    }
                }
            }]
        }
    
    def deploy_azure_container_instances(self) -> bool:
        """Deploy to Azure Container Instances"""
        self.log("Deploying to Azure Container Instances...")
        
        try:
            # Check Azure CLI
            self.run_command("az --version")
            
            # Create resource group
            resource_group = f"divine-agents-{self.deployment_id}"
            self.run_command(
                f"az group create --name {resource_group} --location eastus"
            )
            self.log(f"✓ Resource group {resource_group} created")
            
            # Create container instance
            self.run_command(
                f"az container create "
                f"--resource-group {resource_group} "
                f"--name divine-agent-system "
                f"--image divine-agent-system:latest "
                f"--cpu 2 "
                f"--memory 4 "
                f"--ports 8000 "
                f"--environment-variables ENVIRONMENT=production QUANTUM_ENABLED=true"
            )
            self.log("✓ Azure Container Instance created")
            
            return True
            
        except subprocess.CalledProcessError:
            self.log("Failed to deploy to Azure Container Instances", "ERROR")
            return False
    
    def deploy_gcp_cloud_run(self) -> bool:
        """Deploy to Google Cloud Run"""
        self.log("Deploying to Google Cloud Run...")
        
        try:
            # Check gcloud CLI
            self.run_command("gcloud --version")
            
            # Build and push to Container Registry
            project_id = self.run_command("gcloud config get-value project").stdout.strip()
            image_url = f"gcr.io/{project_id}/divine-agent-system:latest"
            
            self.run_command(f"docker tag divine-agent-system:latest {image_url}")
            self.run_command(f"docker push {image_url}")
            self.log("✓ Image pushed to Container Registry")
            
            # Deploy to Cloud Run
            self.run_command(
                f"gcloud run deploy divine-agent-system "
                f"--image {image_url} "
                f"--platform managed "
                f"--region us-central1 "
                f"--allow-unauthenticated "
                f"--set-env-vars ENVIRONMENT=production,QUANTUM_ENABLED=true"
            )
            self.log("✓ Deployed to Google Cloud Run")
            
            return True
            
        except subprocess.CalledProcessError:
            self.log("Failed to deploy to Google Cloud Run", "ERROR")
            return False
    
    def run_post_deployment_tests(self) -> bool:
        """Run post-deployment validation tests"""
        self.log("Running post-deployment tests...")
        
        try:
            # Run system tests
            result = self.run_command("python3 test_system.py", check=False)
            if result.returncode == 0:
                self.log("✓ System tests passed")
                return True
            else:
                self.log("System tests failed", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Failed to run post-deployment tests: {e}", "ERROR")
            return False
    
    def cleanup_deployment(self, platform: str):
        """Clean up deployment resources"""
        self.log(f"Cleaning up {platform} deployment...")
        
        try:
            if platform == "docker":
                self.run_command("docker-compose down")
            elif platform == "kubernetes":
                self.run_command("kubectl delete namespace divine-agents")
            elif platform == "aws":
                cluster_name = f"divine-agents-{self.deployment_id}"
                self.run_command(f"aws ecs delete-cluster --cluster {cluster_name}")
            elif platform == "azure":
                resource_group = f"divine-agents-{self.deployment_id}"
                self.run_command(f"az group delete --name {resource_group} --yes")
            
            self.log(f"✓ {platform} resources cleaned up")
            
        except subprocess.CalledProcessError:
            self.log(f"Failed to cleanup {platform} resources", "ERROR")
    
    def deploy(self, platform: str, **kwargs) -> bool:
        """Main deployment orchestration"""
        self.log(f"Starting Divine Agent System deployment to {platform}")
        self.log(f"Deployment ID: {self.deployment_id}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Build images
        if not self.build_docker_images():
            return False
        
        # Deploy to target platform
        success = False
        if platform == "docker":
            success = self.deploy_local_docker()
        elif platform == "kubernetes":
            success = self.deploy_kubernetes(kwargs.get('namespace', 'divine-agents'))
        elif platform == "aws":
            success = self.deploy_aws_ecs()
        elif platform == "azure":
            success = self.deploy_azure_container_instances()
        elif platform == "gcp":
            success = self.deploy_gcp_cloud_run()
        else:
            self.log(f"Unsupported platform: {platform}", "ERROR")
            return False
        
        if success:
            # Run post-deployment tests
            if self.run_post_deployment_tests():
                self.log("🎉 Deployment completed successfully!")
                self.log(f"Divine Agent System is now running on {platform}")
                self.log(f"Deployment log: {self.log_file}")
                return True
            else:
                self.log("Deployment completed but tests failed", "WARNING")
                return False
        else:
            self.log("Deployment failed", "ERROR")
            return False

def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(
        description="Divine Agent System Deployment Orchestrator"
    )
    parser.add_argument(
        'platform',
        choices=['docker', 'kubernetes', 'aws', 'azure', 'gcp'],
        help='Deployment platform'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--namespace',
        default='divine-agents',
        help='Kubernetes namespace (for k8s deployments)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up deployment resources'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run tests only (no deployment)'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DivineDeploymentOrchestrator(args.config)
    
    if args.test_only:
        # Run tests only
        success = orchestrator.run_post_deployment_tests()
        sys.exit(0 if success else 1)
    
    if args.cleanup:
        # Clean up resources
        orchestrator.cleanup_deployment(args.platform)
        sys.exit(0)
    
    # Deploy system
    success = orchestrator.deploy(
        args.platform,
        namespace=args.namespace
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()