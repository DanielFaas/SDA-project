# Statistical Analysis Project

This project is set up for  statistical analysis using Python 3.12, uv and Docker.

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
*   **File Sync:** The container uses a "bind mount". Files created inside the container (like `Plots/result.png`) will instantly appear in your OS folder, and vice-versa.

---

## Running without VS Code (CI/CLI)

If you need to run this without VS Code:

1.  **Build the image:**
    ```bash
    docker build -t stats-project .
    ```
2.  **Run the script:**
    ```bash
    docker run --rm -v $(pwd):/app stats-project python main.py
    ```
```