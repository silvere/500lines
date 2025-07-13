from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="langsmith-agent-sdk",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="一个强大而易用的LangSmith SDK，专为Agent的trace追踪和evaluation评估而设计",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/langsmith-agent-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "jupyter>=1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "langsmith-agent-sdk=langsmith_agent_sdk.cli:main",
        ],
    },
    keywords="langsmith, agent, tracing, evaluation, ai, llm",
    project_urls={
        "Documentation": "https://github.com/yourusername/langsmith-agent-sdk",
        "Source": "https://github.com/yourusername/langsmith-agent-sdk",
        "Tracker": "https://github.com/yourusername/langsmith-agent-sdk/issues",
    },
)