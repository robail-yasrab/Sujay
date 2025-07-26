#!/usr/bin/env python3
"""
Script to help push the Video Deepfake Detection project to GitHub
Username: robail-yasrab
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"   Error: {e.stderr}")
        return False

def check_git_installed():
    """Check if Git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("🚀 GitHub Push Helper for Video Deepfake Detection Project")
    print("=" * 60)
    print(f"👤 GitHub Username: robail-yasrab")
    print()

    # Check if Git is installed
    if not check_git_installed():
        print("❌ Git is not installed or not found in PATH")
        print("Please install Git from: https://git-scm.com/downloads")
        return

    # Repository details
    repo_name = "video-deepfake-detection"
    github_username = "robail-yasrab"
    repo_url = f"https://github.com/{github_username}/{repo_name}.git"
    
    print(f"📁 Repository Name: {repo_name}")
    print(f"🔗 Repository URL: {repo_url}")
    print()

    # Step 1: Initialize Git repository
    if not os.path.exists(".git"):
        if not run_command("git init", "Initializing Git repository"):
            return
    else:
        print("✅ Git repository already initialized")

    # Step 2: Configure Git user (if not already configured)
    print("\n🔧 Configuring Git user...")
    run_command(f'git config user.name "robail-yasrab"', "Setting Git username")
    run_command(f'git config user.email "your-email@example.com"', "Setting Git email (please update this)")

    # Step 3: Add all files
    if not run_command("git add .", "Adding all files to Git"):
        return

    # Step 4: Check status
    print("\n📋 Current Git status:")
    run_command("git status", "Checking Git status")

    # Step 5: Commit changes
    commit_message = "Initial commit: Video Deepfake Detection using Projective Geometry"
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        print("ℹ️  Note: If no changes to commit, that's normal if you've already committed.")

    # Step 6: Add remote origin
    print(f"\n🔗 Adding remote origin: {repo_url}")
    run_command(f"git remote remove origin", "Removing existing origin (if any)")
    if not run_command(f"git remote add origin {repo_url}", "Adding remote origin"):
        return

    # Step 7: Create main branch and push
    print("\n🚀 Pushing to GitHub...")
    if not run_command("git branch -M main", "Setting main branch"):
        return
    
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        print("\n❌ Push failed. This might be because:")
        print("1. The repository doesn't exist on GitHub yet")
        print("2. Authentication issues")
        print("3. Network connectivity problems")
        print("\nNext steps:")
        print(f"1. Go to: https://github.com/{github_username}")
        print("2. Click 'New repository' or go to: https://github.com/new")
        print(f"3. Create a repository named: {repo_name}")
        print("4. Run this script again")
        return

    print("\n🎉 SUCCESS! Your code has been pushed to GitHub!")
    print(f"🔗 View your repository at: https://github.com/{github_username}/{repo_name}")
    
    # Step 8: Create .gitignore if it doesn't exist
    create_gitignore()

def create_gitignore():
    """Create a .gitignore file for Python projects"""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Video files (optional - uncomment if you want to exclude test videos)
# *.mp4
# *.avi
# *.mov
# test_videos/

# IDE
.vscode/
.idea/
*.swp
*.swo
"""
    
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")

if __name__ == "__main__":
    main() 