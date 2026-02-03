# Summarization-CNN-DailyMail

![Text Summarization](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=professional%20text%20summarization%20application%20interface%20with%20blue%20color%20scheme%20and%20modern%20design&image_size=landscape_16_9)

## Project Overview

A professional-grade AI-powered text summarization application specifically designed for CNN DailyMail articles. This project combines advanced extractive and abstractive summarization techniques to provide high-quality summaries with an intuitive, modern user interface.

### Key Accomplishments

- **Dual Summarization Techniques**: Implements both extractive and abstractive summarization methods
- **State-of-the-Art Models**: Utilizes pre-trained models for optimal performance
- **Professional UI/UX**: Modern, responsive design with intuitive user experience
- **Efficient Processing**: Optimized model loading and inference for faster results
- **Comprehensive Documentation**: Detailed setup and usage instructions

## Features & Functionalities

### Core Features

- **Extractive Summarization**: Identifies and extracts key sentences from original text using SentenceTransformer embeddings and KMeans clustering
- **Abstractive Summarization**: Generates new, coherent summaries using a fine-tuned T5 model with LoRA adapter
- **Real-time Processing**: Provides instant feedback with loading states and processing metrics
- **Input Validation**: Enforces 512-token limit (≈2048 characters) with visual indicators
- **Detailed Metrics**: Displays processing time and text length statistics
- **Responsive Design**: Adapts to different screen sizes for cross-device compatibility

### Technical Implementation

- **Backend**: FastAPI server with model loading during initialization
- **Frontend**: Modern HTML/CSS/JavaScript interface with no external dependencies
- **Model Management**: Efficient loading and caching of pre-trained models
- **Error Handling**: Comprehensive error management with user-friendly messages
- **Logging**: Detailed logging for debugging and monitoring

## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- Pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/GregReynaldi/Summarization-CNN-DailyMail.git
cd Summarization-CNN-DailyMail
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Pre-trained Models

The project requires two pre-trained models:

1. **Extractive Model**: SentenceTransformer model trained for sentence embeddings
2. **Abstractive Model**: Fine-tuned T5 model with LoRA adapter

These models should be placed in the following directories:

- `modelExtractive/`
- `modelAbstractive/`

### Step 4: Dataset Information (Optional)

The CNN DailyMail dataset used for training the models is not included in this repository due to its large size. If you need the dataset for training or evaluation purposes, you can download it from:

[Kaggle: Newspaper Text Summarization CNN-DailyMail](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

### Step 5: Start the Backend Server

```bash
python -m uvicorn application.backend.main:app --reload
```

The server will start on `http://localhost:8000`

### Step 6: Access the Application

Open the frontend interface by navigating to:

```
application/frontend/index.html
```

Double-click the file to open it in your default web browser.

## Usage Guidelines

### Basic Usage

1. **Enter Text**: Paste your CNN DailyMail article text into the input area
2. **Select Summarization Type**:
   - **Extractive**: Extracts key sentences from the original text
   - **Abstractive**: Generates a new summary that captures the essence of the text
3. **Generate Summary**: Click the "Generate Summary" button
4. **View Results**: The summary will appear in the results section with processing metrics

### Example Workflow

1. **Input**:
   ```
   CNN reports that President Biden signed a new executive order on Tuesday aimed at strengthening America's cybersecurity defense capabilities. The order requires federal agencies to implement stricter cybersecurity standards within the next 180 days and encourages private businesses to take similar measures.
   
   Biden stated during the signing ceremony: "Cybersecurity threats are among the most urgent challenges facing our nation, and we must take all necessary measures to protect our critical infrastructure and data."
   
   The order also includes the establishment of a new cybersecurity coordination committee responsible for overseeing nationwide cybersecurity efforts and enhancing cooperation with international allies.
   ```

2. **Output** (Abstractive):
   ```
   President Biden signed a new executive order to strengthen U.S. cybersecurity defenses, requiring federal agencies to implement stricter standards within 180 days and encouraging private businesses to do the same. He emphasized that cybersecurity threats are among the nation's most urgent challenges. The order also establishes a new cybersecurity coordination committee to oversee national efforts and enhance international cooperation.
   ```

## Configuration Options

### Backend Configuration

The backend server can be configured through environment variables:

| Environment Variable | Description | Default Value |
|----------------------|-------------|---------------|
| `HOST` | Host address for the server | `0.0.0.0` |
| `PORT` | Port for the server | `8000` |
| `DEBUG` | Enable debug mode | `False` |

### Model Configuration

Model paths are configured in the `main.py` file:

```python
# Extractive model path
extractive_model_path = os.path.join(BASE_PATH, "modelExtractive")

# Abstractive model path
abstractive_model_path = os.path.join(BASE_PATH, "modelAbstractive")
```

### Frontend Configuration

The frontend can be customized by modifying the CSS variables in `index.html`:

```css
:root {
    --primary-color: #1a56db;
    --primary-dark: #1e40af;
    --secondary-color: #0ea5e9;
    /* Additional variables */
}
```

## API Reference

### Summarization Endpoint

**POST /summarize**

Request body:

```json
{
  "text": "Article text to summarize",
  "type": "extractive" // or "abstractive"
}
```

Response:

```json
{
  "summary": "Generated summary",
  "type": "extractive",
  "processing_time": 1.23,
  "input_length": 500,
  "output_length": 150
}
```

### Health Check Endpoint

**GET /health**

Response:

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Contribution Guidelines

### How to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit Changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to Branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use 4 spaces for indentation
- Write clear, concise docstrings for all functions
- Include type hints for function parameters and return values

### Testing

Contributions should include tests for any new functionality:

```bash
# Run existing tests
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hugging Face Transformers**: For providing the pre-trained models and pipeline API
- **Sentence Transformers**: For the extractive summarization implementation
- **FastAPI**: For the high-performance backend server
- **CNN DailyMail Dataset**: For the training data used to fine-tune the models

## Contact

### Project Maintainers

- **Gregorius Reynaldi**
  - Email: gregoriusreynaldi@gmail.com
  - GitHub: [GregReynaldi](https://github.com/GregReynaldi)

### Support

For questions, issues, or feature requests:

1. **GitHub Issues**: [https://github.com/GregReynaldi/Summarization-CNN-DailyMail/issues](https://github.com/GregReynaldi/Summarization-CNN-DailyMail/issues)
2. **Email**: gregoriusreynaldi@gmail.com

## Project Status

### Current Version
v1.0.0

### Key Milestones Achieved

- ✅ Implementation of extractive summarization using SentenceTransformer
- ✅ Implementation of abstractive summarization using fine-tuned T5 model
- ✅ Development of professional, responsive frontend interface
- ✅ Integration of both summarization techniques into a unified application
- ✅ Comprehensive error handling and input validation
- ✅ Detailed documentation and usage guidelines

### Future Roadmap

- [ ] Add support for additional summarization models
- [ ] Implement batch processing for multiple documents
- [ ] Add export functionality for summaries
- [ ] Develop browser extension for easier access
- [ ] Improve model performance and accuracy

---

*This project was developed with a focus on professional standards for both functionality and design, providing a reliable and user-friendly solution for text summarization tasks.*
