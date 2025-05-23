#!/usr/bin/env python3

import talon
talon.init()

from talon.signature import extract

def test_debug_extract():
    msg_body = 'Blah\r\n--\r\n\r\nSergey Obukhov'
    result = extract(msg_body, 'Sergey')
    print(f"Input: {repr(msg_body)}")
    print(f"Result: {repr(result)}")
    print(f"Expected: {repr(('Blah', '--\r\n\r\nSergey Obukhov'))}")
    
    # Let's also test what the actual signature boundary detection does
    print("\n--- Detailed analysis ---")
    print(f"Result[0] (text): {repr(result[0])}")
    print(f"Result[1] (signature): {repr(result[1])}")

if __name__ == "__main__":
    test_debug_extract()
