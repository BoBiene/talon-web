talon-web-api
=====

Mailgun library (https://github.com/mailgun/talon) to extract message quotations and signatures hosted as web-api in a docker container.

If you ever tried to parse message quotations or signatures you know that absence of any formatting standards in this area could make this task a nightmare. Hopefully this library will make your life much easier. The name of the project is inspired by TALON - multipurpose robot designed to perform missions ranging from reconnaissance to combat and operate in a number of hostile environments. That’s what a good quotations and signature parser should be like :smile:

## Usage

Talon can be used as a webservice. Can be invoked by using the script.

### Pre-Build Docker-Image
```
docker run -p 5505:5505 ghcr.io/bobiene/talon-web:latest 
```

### From Source
``` 
./run-web.sh
```

Or via docker

```
./build-dock.sh
./run-dock.sh
```

# API


## Endpoints
- `/health` (Health check)
- `/talon/signature`
- `/talon/quotations/text`
- `/talon/quotations/html`
- `/talon/html-to-markdown`
- `/talon/html-to-markdown-direct`


### Endpoint `/health` ``GET``

Health check endpoint for monitoring and load balancers.

#### Response

```json
{
    "status": "healthy",
    "service": "talon-web-api", 
    "version": "1.6.0",
    "endpoints": [
        "/talon/signature",
        "/talon/quotations/text",
        "/talon/quotations/html", 
        "/talon/html-to-markdown",
        "/talon/html-to-markdown-direct"
    ]
}
```

### Endpoint `/talon/signature` ``POST``
| Post-Parameter | provision | comment |
| --- | --- | ---- |
| email_content | *requiered* | plain text of the e-mail body |
| email_sender | *requiered* | e-mail address of the sender |

#### Response

````json
{
    "email_content": "<<content-of-post-parameter email_content>>",
    "email_sender": "<<content-of-post-parameter email_sender>>",
    "email_body": "<<striped-e-mail-text (without signature)>>",
    "email_signature": "<<signature, if found>>|None"
}
````

### Endpoint `/talon/quotations/text` ``POST``
| Post-Parameter | provision | comment |
| --- | --- | ---- |
| email_content | *requiered* | plain text of the e-mail body |
| email_sender | *optional* | e-mail address of the sender, if provided not only the quotation is stripped of but also the signature if found |

#### Response

*without* `email_sender`
````json
{
    "email_content": "<<content-of-post-parameter email_content>>",
    "email_reply": "<<striped-e-mail-text>>"
}
````

*with* `email_sender`
````json
{
    "email_content": "<<content-of-post-parameter email_content>>",
    "email_sender": "<<content-of-post-parameter email_sender>>",
    "email_reply": "<<striped-e-mail-text (without signature)>>",
    "email_signature": "<<signature, if found>>|None"
}
````

#### Endpoint `/talon/quotations/html` ``POST``

| Post-Parameter | provision | comment |
| --- | --- | ---- |
| email_content | *requiered* | HTML of the e-mail body |
| email_sender | *optional* | e-mail address of the sender, if provided not only the quotation is stripped of but also the signature if found |

#### Response

*without* `email_sender`
````json
{
    "email_content": "<<content-of-post-parameter email_content>>",
    "email_reply": "<<striped-e-mail-text>>"
}
````

*with* `email_sender`
````json
{
    "email_content": "<<content-of-post-parameter email_content>>",
    "email_sender": "<<content-of-post-parameter email_sender>>",
    "email_reply": "<<striped-e-mail-text (without signature)>>",
    "email_signature": "<<signature, if found>>|None"
}
````

Sample
------
For endpoint `/talon/signature`, invoked as a `get` or `post` request. Curl Sample:

```
curl --location --request GET 'http://127.0.0.1:5505/talon/signature' \
--form 'email_content="Hi,

This is just a test.

Thanks,
John Doe
mobile: 052543453
email: john.doe@anywebsite.ph
website: www.anywebsite.ph"' \
--form 'email_sender="John Doe . . <john.doe@anywebsite.ph>"'
```

You will be required to pass a body of type *form-data* as a parameter.
Keys are `email_content` and `email_sender`.

Response will include `email_signature`. Sample response below:

```
{
    "email_content": "Hi,\n\nThis is just a test.\n\nThanks,\nJohn Doe\nmobile: 052543453\nemail: john.doe@anywebsite.ph\nwebsite: www.anywebsite.ph",
    "email_sender": "John Doe . . <john.doe@anywebsite.ph>",
    "email_signature": "Thanks,\nJohn Doe\nmobile: 052543453\nemail: john.doe@anywebsite.ph\nwebsite: www.anywebsite.ph"
}

```



Research
--------

The library is inspired by the following research papers and projects:

-  http://www.cs.cmu.edu/~vitor/papers/sigFilePaper_finalversion.pdf
-  http://www.cs.cornell.edu/people/tj/publications/joachims_01a.pdf

## HTML to Markdown Endpoints

### Endpoint `/talon/html-to-markdown` ``POST``

Converts HTML to Markdown using Talon's intelligent signature and quotation detection combined with html2text.

| Post-Parameter | provision | comment |
| --- | --- | ---- |
| html | *requiered* | HTML content to be converted |
| sender | *optional* | sender's email address for enhanced signature detection |

#### Request Format

**JSON Input:**
```json
{
    "html": "<html><body><h1>Title</h1><p>Content...</p><hr><p>Best regards...</p></body></html>",
    "sender": "max@example.com"
}
```

**Alternative Form-Data Input:**
- `html_content`: HTML content
- `email_sender`: sender email (optional)

#### Response

```json
{
    "original_html": "<<content-of-post-parameter html>>",
    "markdown": "# Title\n\nContent...",
    "removed_signature": "Best regards\nMax Mustermann",
    "sender": "max@example.com",
    "success": true
}
```

### Endpoint `/talon/html-to-markdown-direct` ``POST``

Direct HTML to Markdown conversion with basic signature pattern recognition, without Talon's quotation extraction.

| Post-Parameter | provision | comment |
| --- | --- | ---- |
| html | *requiered* | HTML content to be converted |

#### Request Format

**JSON Input:**
```json
{
    "html": "<html><body><h1>Title</h1><p>Content...</p><hr><p>Best regards...</p></body></html>"
}
```

**Alternative Form-Data Input:**
- `html_content`: HTML content

#### Response

```json
{
    "original_html": "<<content-of-post-parameter html>>",
    "markdown": "# Title\n\nContent...",
    "success": true
}
```

### Curl Examples

**With Talon's intelligent detection:**
```bash
curl -X POST 'http://127.0.0.1:5505/talon/html-to-markdown' \
--header 'Content-Type: application/json' \
--data '{
    "html": "<h1>Test</h1><p>Important content</p><hr><p>Best regards<br>Max Mustermann</p>",
    "sender": "max@example.com"
}'
```

**Direct conversion:**
```bash
curl -X POST 'http://127.0.0.1:5505/talon/html-to-markdown-direct' \
--header 'Content-Type: application/json' \
--data '{
    "html": "<h1>Test</h1><p>Important content</p><hr><p>Best regards<br>Max Mustermann</p>"
}'
```

### Recognized Signature Patterns

The HTML-to-Markdown endpoints recognize the following signature patterns:

**German Patterns:**
- "Mit freundlichen Grüßen"
- "Freundliche Grüße"  
- "Viele Grüße"

**English Patterns:**
- "Best regards"
- "Kind regards"
- "Sincerely"

**Technical Patterns:**
- `<hr>` tags (everything after the tag is removed)
- `--` separators
- CSS classes containing "signature"
- Gmail/Outlook signature blocks

### Features

- **HTML-to-Markdown Conversion**: Uses html2text for clean Markdown output
- **Intelligent Signature Removal**: Combines Talon's ML-based detection with pattern matching
- **Flexible Input**: Supports both JSON and form-data inputs
- **Multilingual Support**: Recognizes German and English signature patterns
- **AI-Powered Image Processing**: Download and describe images using OpenAI Vision API
- **Two Conversion Modes**: 
  - Intelligent with Talon's quotation extraction
  - Direct with simple pattern recognition

### AI Image Processing

The `/talon/html-to-markdown` endpoint supports automatic image processing with AI-generated descriptions:

#### Optional Parameters for AI Features

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `openai_api_key` | string | OpenAI API Key for image descriptions | `null` (disabled) |
| `base_url` | string | Base URL for resolving relative image URLs | `null` |
| `image_path` | string | Local directory for downloaded images | `"./images/"` |
| `image_prefix` | string | Prefix for downloaded image filenames | `""` |

#### How It Works

1. **Image Detection**: Finds all `<img>` tags in HTML
2. **URL Resolution**: Converts relative URLs to absolute using `base_url`
3. **Image Download**: Downloads images to local `image_path`
4. **AI Description**: Uses OpenAI Vision API to generate German descriptions
5. **Markdown Enhancement**: Replaces images with enhanced format

#### Enhanced Markdown Format

Images are converted to this enhanced format:

```markdown
![VW ID.7 Tourer im Schnee](./images/vw-id7.jpg)

> **Bildbeschreibung (KI):**
> Ein roter VW ID.7 Tourer steht im Schnee vor einem Einfamilienhaus. Im Hintergrund sind verschneite Bäume und ein bewölkter Himmel zu sehen.
```

#### Example Request with AI Processing

```bash
curl -X POST 'http://127.0.0.1:5505/talon/html-to-markdown' \
--header 'Content-Type: application/json' \
--data '{
    "html": "<h1>Newsletter</h1><p>Unser neues Auto:</p><img src=\"car.jpg\" alt=\"Neues Fahrzeug\" />",
    "sender": "marketing@auto.de",
    "openai_api_key": "sk-...",
    "base_url": "https://company.com/newsletter/",
    "image_path": "./downloads/newsletter/",
    "image_prefix": "auto-2024-"
}'
```

#### Example Response with AI Processing

```json
{
    "markdown": "# Newsletter\n\nUnser neues Auto:\n\n![Neues Fahrzeug](./downloads/newsletter/auto-2024-abc123.jpg)\n\n> **Bildbeschreibung (KI):**\n> Ein silberner SUV steht in einer modernen Ausstellungshalle mit Glasfront und Beleuchtung.",
    "processed_images": ["car.jpg"],
    "downloaded_images": ["./downloads/newsletter/auto-2024-abc123.jpg"],
    "original_html": "...",
    "email_sender": "marketing@auto.de"
}
```

#### Cost Optimization

- Uses `gpt-4.1-mini` model for cost efficiency
- Images processed at low detail level
- Only processes images when `openai_api_key` is provided
- Skips data URLs (base64 embedded images)
