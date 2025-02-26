# Flask Quick Start Example

This repository contains a [Flask](http://flask.pocoo.org/) application example based on the official [Flask Quick Start Guide](http://flask.pocoo.org/docs/1.0/quickstart/#a-minimal-application), configured for deployment on [Render](https://render.com).

🌐 **Live Demo**: [https://flask.onrender.com](https://flask.onrender.com)

## 🚀 Deployment Setup

For detailed deployment instructions, please refer to the [Render Flask Deployment Guide](https://render.com/docs/deploy-flask).

## 💻 Development Setup

### Prerequisites

- Ensure you have Python installed on your system
- Basic understanding of command line operations

### Installation Steps

1. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**
   
   Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   
   macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

   When successfully activated, your prompt should change to:
   ```
   (venv) C:\your\current\directory>
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   
   Standard mode:
   ```bash
   flask --app app run
   ```
   
   Debug mode:
   ```bash
   flask --app app run --debug
   ```

### ⚠️ Important Notes

- If Flask is not found after installation, you may need to add it to your system's PATH.
- Locate the Flask executable in your virtual environment and add its path to your system's PATH variable.
- Ensure Python is being executed from the correct location if you experience path-related issues.

### 🔍 Development Server Warning

```
⚠️ WARNING: This is a development server.
Do not use it in a production deployment.
Use a production WSGI server instead.
```

## 📝 License

[MIT License](LICENSE)

---

<details>
<summary>Troubleshooting Tips</summary>

- If you encounter PATH issues, verify your Python installation location
- Make sure your virtual environment is activated before running any Flask commands
- Check that all dependencies are properly installed using `pip list`
</details>