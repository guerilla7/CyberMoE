from preprocessor import CyberPreprocessor

def test_preprocessor():
    # Initialize preprocessor
    preprocessor = CyberPreprocessor()
    
    # Test text with various technical features
    test_text = """
    Suspicious network traffic detected from IP 192.168.1.100 to malicious domain evil.com on port 443/tcp.
    The attacker attempted SQL injection at https://target.com/login.php.
    Malware hash: 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8
    CVE-2021-44228 vulnerability exploited in the system.
    """
    
    # Process the text
    result = preprocessor.process(test_text)
    
    # Verify features were extracted
    assert len(result['features']['ips']) > 0, "Failed to extract IP"
    assert len(result['features']['urls']) > 0, "Failed to extract URL"
    assert len(result['features']['ports']) > 0, "Failed to extract port"
    assert len(result['features']['hashes']) > 0, "Failed to extract hash"
    assert len(result['features']['cves']) > 0, "Failed to extract CVE"
    assert len(result['features']['domains']) > 0, "Failed to extract domain"
    
    # Verify domain scores
    assert len(result['domain_scores']) == 5, "Incorrect number of domain scores"
    assert result['domain_scores'].max() <= 1.0, "Domain score exceeds 1.0"
    assert result['domain_scores'].min() >= 0.0, "Domain score below 0.0"
    
    # Verify entity types
    assert result['entity_types'].shape == (len(result['tokens']),), "Entity types shape mismatch"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_preprocessor()
