talon-web-api
=====

Mailgun library (https://github.com/mailgun/talon) to extract message quotations and signatures hosted as web-api in a docker container.

If you ever tried to parse message quotations or signatures you know that absence of any formatting standards in this area could make this task a nightmare. Hopefully this library will make your life much easier. The name of the project is inspired by TALON - multipurpose robot designed to perform missions ranging from reconnaissance to combat and operate in a number of hostile environments. That’s what a good quotations and signature parser should be like :smile:

Usage
-----

Talon can be used as a webservice. Can be invoked by using the script.



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
- `/talon/signature`
- `/talon/quotations/text`
- `/talon/quotations/html`


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
curl --location --request GET 'http://127.0.0.1:5000/talon/signature' \
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
