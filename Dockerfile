# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create the user FIRST
RUN useradd -m -u 1000 user

# Copy requirements and install dependencies
# We copy as 'user' to ensure permissions are right from the start
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and Change Ownership to 'user'
COPY --chown=user . .

# Switch to the non-root user
USER user

# Expose port 7860
EXPOSE 7860

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]