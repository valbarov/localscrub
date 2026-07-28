"""Shared fixtures.

Test data policy
----------------
Every identifier in this test suite is synthetic, drawn from ranges reserved
for documentation and testing:

- phone numbers: 555-01XX (reserved fictional range)
- emails/URLs: example.com / example.org (RFC 2606)
- IP addresses: 192.0.2.x, 198.51.100.x, 203.0.113.x (RFC 5737)
- card numbers: industry test numbers (e.g. 4111 1111 1111 1111)
- SSNs: the canonical invalid example 123-45-6789 and structurally
  invalid values for negative tests
- MRNs/accounts/plan IDs: invented values

No text from any clinical source, no names of real people. Ever.
"""
