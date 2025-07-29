#!/usr/bin/env python3
"""
Divine Agent System - Supreme Agentic Orchestrator (SAO)
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file"""
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out comments, empty lines, and built-in modules
        requirements = []
        builtin_modules = {
            'asyncio', 'os', 'sys', 'pathlib', 'uuid', 'statistics', 
            'math', 'random', 'concurrent.futures', 'multiprocessing', 
            'threading', 'sqlite3'
        }
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before >= or ==)
                package_name = line.split('>=')[0].split('==')[0].split('[')[0]
                if package_name not in builtin_modules:
                    requirements.append(line)
        
        return requirements
    return []

# Version information
VERSION = "1.0.0"
AUTHOR = "Divine Agent System Team"
AUTHOR_EMAIL = "contact@divineagentsystem.ai"
DESCRIPTION = "Supreme Agentic Orchestrator - Multi-Agent Cloud Mastery System"
URL = "https://github.com/divineagentsystem/sao"

# Classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: System Administrators",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Distributed Computing",
    "Topic :: System :: Monitoring",
    "Topic :: System :: Systems Administration",
    "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Security",
    "Framework :: AsyncIO",
]

# Keywords
KEYWORDS = [
    "agents", "multi-agent", "orchestrator", "cloud", "devops", 
    "kubernetes", "serverless", "security", "monitoring", "cost-optimization",
    "data-engineering", "quantum", "consciousness", "ai", "automation",
    "infrastructure", "microservices", "containers", "observability"
]

# Entry points for command-line tools
ENTRY_POINTS = {
    'console_scripts': [
        'sao=agents.cli:main',
        'divine-agent=agents.cli:main',
        'supreme-orchestrator=agents.cli:main',
    ],
}

# Package data
PACKAGE_DATA = {
    'agents': [
        '*.md',
        '*.txt',
        '*.yaml',
        '*.yml',
        '*.json',
        'config/*',
        'templates/*',
        'schemas/*',
    ],
}

# Extra requirements for different use cases
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.1.0',
        'pytest-mock>=3.11.0',
        'black>=23.7.0',
        'flake8>=6.0.0',
        'mypy>=1.5.0',
        'isort>=5.12.0',
        'sphinx>=7.1.0',
        'sphinx-rtd-theme>=1.3.0',
    ],
    'quantum': [
        'qiskit>=0.44.0',
        'cirq>=1.2.0',
        'qiskit-aer>=0.12.0',
    ],
    'ml': [
        'scikit-learn>=1.3.0',
        'tensorflow>=2.13.0',
        'torch>=2.0.0',
        'transformers>=4.32.0',
    ],
    'nlp': [
        'nltk>=3.8.0',
        'spacy>=3.6.0',
        'transformers>=4.32.0',
    ],
    'cloud-aws': [
        'boto3>=1.28.0',
        'botocore>=1.31.0',
    ],
    'cloud-azure': [
        'azure-storage-blob>=12.17.0',
        'azure-identity>=1.14.0',
    ],
    'cloud-gcp': [
        'google-cloud-storage>=2.10.0',
        'google-auth>=2.22.0',
    ],
    'monitoring': [
        'prometheus-client>=0.17.0',
        'grafana-api>=1.0.3',
        'influxdb-client>=1.37.0',
    ],
    'messaging': [
        'redis>=4.6.0',
        'pika>=1.3.0',
        'kafka-python>=2.0.2',
    ],
    'all': [
        # This will include all optional dependencies
        'qiskit>=0.44.0',
        'cirq>=1.2.0',
        'scikit-learn>=1.3.0',
        'tensorflow>=2.13.0',
        'torch>=2.0.0',
        'nltk>=3.8.0',
        'spacy>=3.6.0',
        'boto3>=1.28.0',
        'azure-storage-blob>=12.17.0',
        'google-cloud-storage>=2.10.0',
        'prometheus-client>=0.17.0',
        'redis>=4.6.0',
    ],
}

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Project URLs
PROJECT_URLS = {
    "Bug Reports": "https://github.com/divineagentsystem/sao/issues",
    "Source": "https://github.com/divineagentsystem/sao",
    "Documentation": "https://divineagentsystem.readthedocs.io/",
    "Funding": "https://github.com/sponsors/divineagentsystem",
    "Say Thanks!": "https://saythanks.io/to/divineagentsystem",
}

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Divine Agent System requires Python 3.8 or higher.")
        print(f"You are using Python {sys.version}")
        sys.exit(1)

def main():
    """Main setup function"""
    check_python_version()
    
    setup(
        name="divine-agent-system",
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url=URL,
        project_urls=PROJECT_URLS,
        packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
        package_data=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
        keywords=" ".join(KEYWORDS),
        python_requires=PYTHON_REQUIRES,
        install_requires=read_requirements(),
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        include_package_data=True,
        zip_safe=False,
        platforms=["any"],
        license="MIT",
        
        # Additional metadata
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
        download_url=f"{URL}/archive/v{VERSION}.tar.gz",
        
        # Options for development
        options={
            'build_scripts': {
                'executable': '/usr/bin/env python3',
            },
        },
        
        # Test suite
        test_suite="tests",
        tests_require=EXTRAS_REQUIRE['dev'],
        
        # Command line interface
        scripts=[],  # We use entry_points instead
    )

if __name__ == "__main__":
    main()