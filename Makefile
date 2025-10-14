# ----------- CONFIG -----------
VENV_DIR = venv
REQ_FILE = requirements.txt

# --- Кроссплатформенные команды ---
ifeq ($(OS),Windows_NT)
	ACTIVATE = $(VENV_DIR)\Scripts\activate
	PYTHON = $(VENV_DIR)\Scripts\python.exe
	PIP = $(VENV_DIR)\Scripts\pip.exe
else
	ACTIVATE = source $(VENV_DIR)/bin/activate
	PYTHON = $(VENV_DIR)/bin/python
	PIP = $(VENV_DIR)/bin/pip
endif

# ----------- COMMANDS -----------

# Создание виртуального окружения
venv:
	python -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created in $(VENV_DIR)"

# Установка зависимостей
install:
	$(PIP) install --upgrade pip
	$(PIP) install numpy gym opencv-python scipy torch torchvision tensorflow
	@echo "✅ Project dependencies installed"

# Обновление requirements.txt
freeze:
	$(PIP) freeze > $(REQ_FILE)
	@echo "✅ Dependencies frozen to $(REQ_FILE)"

# Полная настройка с нуля
setup: venv install freeze

# Удаление виртуального окружения
clean:
	@echo "🧹 Removing virtual environment..."
ifeq ($(OS),Windows_NT)
	rmdir /s /q $(VENV_DIR)
else
	rm -rf $(VENV_DIR)
endif
	@echo "✅ Done"

# Проверка списка пакетов
list:
	$(PIP) list
