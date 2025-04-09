from setuptools import setup, find_packages

setup(
    name="reflective-learning-agent",
    version="0.1.0",
    description="AI Agent for guided reflection using LangGraph and LLMs",
    author="Your Name",
    author_email="your.email@example.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn>=0.29.0",
        "langchain>=0.2.1",
        "langgraph>=0.1.0",
    ],
) 