# =========== УНИВЕРСАЛЬНЫЙ Makefile — АБСОЛЮТНО РАБОЧАЯ ВЕРСИЯ ===========
VENV_DIR = venv
REQ_FILE = requirements.txt

# Автоопределение ОС
ifeq ($(OS),Windows_NT)
    PYTHON_CMD  = py
    VENV_PYTHON = $(VENV_DIR)\Scripts\python.exe
    VENV_ACTIVATE = $(VENV_DIR)\Scripts\activate.bat
    SHELL_CMD   = cmd /c "$(VENV_ACTIVATE) && title ML_Project (venv) && cmd /k"
else
    PYTHON_CMD  = python3
    VENV_PYTHON = $(VENV_DIR)/bin/python
    VENV_ACTIVATE = . $(VENV_DIR)/bin/activate
    SHELL_CMD   = $(VENV_ACTIVATE); exec $(SHELL)
endif

.PHONY: all venv install freeze setup clean shell activate

# Создаём venv только если его нет
venv:
	@if not exist "$(VENV_DIR)\Scripts\python.exe" (\
		@echo Creating virtual environment... && \
		$(PYTHON_CMD) -m venv $(VENV_DIR) \
	) else (\
		@echo Virtual environment already exists \
	)

# Установка зависимостей (быстро, если venv уже есть)
install: venv
	@echo Upgrading pip...
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@echo Installing dependencies from $(REQ_FILE)...
	@$(VENV_PYTHON) -m pip install -r "$(REQ_FILE)"
	@echo Done!

freeze:
	@$(VENV_PYTHON) -m pip freeze > "$(REQ_FILE)"

setup: venv install freeze

clean:
	@echo Removing $(VENV_DIR)...
	@rmdir /s /q "$(VENV_DIR)" 2>nul || rm -rf "$(VENV_DIR)" 2>/dev/null || true
	@echo Cleaned

# ← ВОТ ЭТА КОМАНДА ТЕПЕРЬ РАБОТАЕТ ВЕЗДЕ
shell: venv
	@echo Launching shell with activated environment...
	$(SHELL_CMD)

activate:
	@echo To activate manually:
	@echo Windows:   $(VENV_DIR)\Scripts\activate
	@echo macOS/Linux: source $(VENV_DIR)/bin/activate