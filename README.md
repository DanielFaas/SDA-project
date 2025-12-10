# Statistical Analysis Project

This project is set up for statistical analysis using Python 3.12, uv and Docker.

## Prerequisites

To work on this project, you need:

1.  **Docker Desktop**
2.  **VS Code**
3.  **Dev Containers Extension** 

---

## Quick Start

1.  Open the project in VS Code.
2.  Open the command palette and type `Dev Containers: Reopen in Container`.
3.  Wait for the container to build.
4.  Once the terminal opens, you are inside the Linux environment.

---

## Managing Packages (uv)

This project uses `uv` instead of standard pip/poetry. All dependency management should happen **inside the VS Code terminal**.

### Add a new library
To add a library (e.g., scipy):
```bash
uv add scipy
```

### Remove a library
```bash
uv remove numpy
```

### Update all packages
```bash
uv sync --upgrade
```

---

## Development Workflow

### Input/Output Data
*   **Data:** Place your raw CSV files in the `Dataset/` folder.
*   **Plots:** Save your figures to the `Plots/` folder.
*   **File Sync:** The container uses a "bind mount". Files created inside the container (like `Plots/result.png`) will instantly appear in your local folder, and vice-versa.

### Important Notes
*   All your project files from your local machine are mounted to `/app` inside the container.
*   Changes made in either the container or your local machine are synchronized automatically.

## Git
1. Start Docker Desktop
2. Make sure you are on main
```bash
git checkout main
```
3. Make sure local repo is up to date
```bash
git pull
```
4. Create and go to your branch
```bash
git checkout -b <branch-name>
```
5. Reopen in container
6. Do your work (commit frequently)
7. Reopen in local
8. Commit with message and add the file you changed (commit regularly)
9. Three dots -> push, pull -> sync (publish the branch if it asks)
10. Repeat 5-9 till the feature the branch targets is finished.
11. When your branch is completely finished create a pull request.


---

## Running without VS Code

If you need to run this without VS Code:

1.  **Build the image:**
    ```bash
    docker build -t stats-project .
    ```
2.  **Run a script:**
    ```bash
    docker run --rm -v $(pwd):/app stats-project python main.py
    ```
3.  **Run an interactive shell:**
    ```bash
    docker run --rm -it -v $(pwd):/app stats-project /bin/bash
    ```

## Combining the datasets
Use the code below to combine two datasets, make sure you change the path if you are in a different folder.
```python
chess_games1 = pd.read_csv("chess_games_risk_part1.csv")

chess_games2 = pd.read_csv("chess_games_risk_part2.csv")


both_games = pd.concat([chess_games1, chess_games2], ignore_index=True, sort=False)
```
