"""
Centralized HTML test data for signature detection tests
"""

# Professional signatures that SHOULD be removed (high confidence)
PROFESSIONAL_SIGNATURES = [
    {
        'name': 'Email_and_job_title',
        'html': '<p>Best regards<br>John Doe<br>Senior Engineer<br>john.doe@company.com</p>',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['john.doe@company.com', 'Senior Engineer']
    },
    {
        'name': 'German_with_email',
        'html': '<div>Mit freundlichen Grüßen<br>Jane Smith<br>Manager<br>jane.smith@example.com</div>',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['jane.smith@example.com', 'Manager']
    },
    {
        'name': 'Multi_language_professional',
        'html': '<p>Schöne Grüße / Best regards<br>Max Mustermann<br>Projektmanager<br>max@company.de</p>',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['max@company.de', 'Projektmanager']
    },
    {
        'name': 'Table_format_signature',
        'html': '''<div>Best regards</div>
        <table><tbody><tr><td>
            <p>Bob Wilson</p>
            <p>Operations Engineer | Customer Solutions</p>
            <p>bob.wilson@example.com | www.example.com</p>
        </td></tr></tbody></table>''',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['bob.wilson@example.com', 'Operations Engineer']
    },
    {
        'name': 'Gmail_signature_class',
        'html': '''<div class="gmail_signature">
            <p>Sarah Connor<br>
            Senior Developer<br>
            sarah@company.com</p>
        </div>''',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['sarah@company.com', 'Senior Developer']
    },
    {
        'name': 'ElementToProof_class',
        'html': '''<div class="elementToProof">
            <p>Schöne Grüße / Best regards<br>
            Anna Weber<br>
            Senior Consultant<br>
            anna.weber@beratung.com</p>
        </div>''',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['anna.weber@beratung.com', 'Senior Consultant']
    }
]

# Simple signatures that should be PRESERVED (conservative approach)
SIMPLE_SIGNATURES = [
    {
        'name': 'Simple_best_regards',
        'html': '<p>Best regards<br>John</p>',
        'expected_removed': False,
        'should_preserve': ['Best regards', 'John'],
        'should_remove': []
    },
    {
        'name': 'German_simple_greeting',
        'html': '<div>Mit freundlichen Grüßen<br>Jane Smith</div>',
        'expected_removed': False,
        'should_preserve': ['Mit freundlichen Grüßen', 'Jane Smith'],
        'should_remove': []
    },
    {
        'name': 'Short_german_greeting',
        'html': '<p>Viele Grüße<br>Max</p>',
        'expected_removed': False,
        'should_preserve': ['Viele Grüße', 'Max'],
        'should_remove': []
    },
    {
        'name': 'Thanks_without_professional',
        'html': '<div>Thanks!<br>Sarah</div>',
        'expected_removed': False,
        'should_preserve': ['Thanks!', 'Sarah'],
        'should_remove': []
    }
]

# Content that should NEVER be removed (salutations in text)
SALUTATION_IN_TEXT = [
    {
        'name': 'Best_in_question',
        'html': '<div>What is the best approach for this problem?</div>',
        'expected_removed': False,
        'should_preserve': ['best approach', 'problem'],
        'should_remove': []
    },
    {
        'name': 'Thanks_in_appreciation',
        'html': '<div>Thanks to your feedback, we improved the system.</div>',
        'expected_removed': False,
        'should_preserve': ['Thanks to your feedback', 'improved'],
        'should_remove': []
    },
    {
        'name': 'Email_in_instructions',
        'html': '<div>Please contact john.doe@example.com for technical support.</div>',
        'expected_removed': False,
        'should_preserve': ['contact john.doe@example.com', 'technical support'],
        'should_remove': []
    },
    {
        'name': 'Regards_in_message',
        'html': '<div>Please send my best regards to your team.</div>',
        'expected_removed': False,
        'should_preserve': ['send my best regards', 'your team'],
        'should_remove': []
    },
    {
        'name': 'Job_title_in_content',
        'html': '<div>Our Senior Engineer will review the proposal.</div>',
        'expected_removed': False,
        'should_preserve': ['Senior Engineer', 'review the proposal'],
        'should_remove': []
    }
]

# Multi-language signature combinations
MULTI_LANGUAGE_SIGNATURES = [
    {
        'name': 'Schoene_gruesse_best_regards_professional',
        'html': '''<div>Schöne Grüße / Best regards<br>
        Max Mustermann<br>
        Senior Project Manager<br>
        max.mustermann@company.com</div>''',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['max.mustermann@company.com', 'Senior Project Manager']
    },
    {
        'name': 'Schoene_gruesse_in_text',
        'html': '<div>Das war super, schöne Grüße / Best regards an dein Team!</div>',
        'expected_removed': False,
        'should_preserve': ['Das war super', 'an dein Team'],
        'should_remove': []
    },
    {
        'name': 'ElementToProof_multi_language',
        'html': '''<div class="elementToProof">
            <p style="margin:0">
                <span>Schöne Grüße / Best regards<br>
                Thomas Schmidt<br>
                thomas.schmidt@firma.de</span>
            </p>
        </div>''',
        'expected_removed': True,
        'should_preserve': [],
        'should_remove': ['thomas.schmidt@firma.de']
    }
]

# Mixed content scenarios (real-world examples)
MIXED_CONTENT_SCENARIOS = [
    {
        'name': 'Content_with_professional_signature',
        'html': '''
        <div>Hello Team,</div>
        <div><br></div>
        <div>Could you please check the server logs for any errors?</div>
        <div>The database shows some connection issues.</div>
        <div><br></div>
        <div>Best regards</div>
        <div>John Doe<br>Senior DevOps Engineer<br>john.doe@company.com</div>
        ''',
        'expected_removed': True,
        'should_preserve': ['Hello Team', 'server logs', 'database shows'],
        'should_remove': ['john.doe@company.com', 'Senior DevOps Engineer']
    },
    {
        'name': 'Content_with_simple_signature',
        'html': '''
        <div>Thanks for the quick response!</div>
        <div>I'll review the documents and get back to you.</div>
        <div><br></div>
        <div>Best regards</div>
        <div>Sarah</div>
        ''',
        'expected_removed': False,
        'should_preserve': ['Thanks for the quick response', 'review the documents', 'Best regards', 'Sarah'],
        'should_remove': []
    }
]

# Edge cases that could cause false positives
EDGE_CASES = [
    {
        'name': 'Data_table_with_email',
        'html': '''<div>Server Details:</div>
        <table>
            <tr><td>Server</td><td>SERVER-123</td></tr>
            <tr><td>Engineer</td><td>John Doe</td></tr>
            <tr><td>Email</td><td>admin@company.com</td></tr>
        </table>''',
        'expected_removed': False,
        'should_preserve': ['Server Details', 'SERVER-123', 'admin@company.com'],
        'should_remove': []
    },
    {
        'name': 'Contact_list',
        'html': '''<div>Contact our team:</div>
        <ul>
            <li>Sales: sales@company.com</li>
            <li>Support: Senior Engineer on duty</li>
            <li>Manager: Available 9-5</li>
        </ul>''',
        'expected_removed': False,
        'should_preserve': ['Contact our team', 'sales@company.com', 'Senior Engineer'],
        'should_remove': []
    },
    {
        'name': 'Quoted_signature',
        'html': '''<div>The email ended with "Best regards, John Doe, Senior Engineer"</div>''',
        'expected_removed': False,
        'should_preserve': ['email ended with', 'Best regards', 'John Doe'],
        'should_remove': []
    }
]
