# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry install

# Command to run tests
CMD ["poetry", "run", "pytest"]