# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in NeuronCite, please report it
privately. **Do not open a public GitHub issue.**

Send an email to: **security@neuroncite.com**

Include the following information:

- Description of the vulnerability
- Steps to reproduce or a proof-of-concept
- Affected version(s)
- Potential impact

## Response Timeline

- **Acknowledgment:** Within 48 hours of receiving your report
- **Assessment:** Initial severity assessment within 5 business days
- **Fix:** Timeline depends on severity:
  - Critical (RCE, authentication bypass): patch within 7 days
  - High (data exposure, privilege escalation): patch within 14 days
  - Medium/Low: addressed in the next scheduled release

## Disclosure Policy

NeuronCite follows coordinated disclosure:

1. The reporter sends the vulnerability details privately.
2. The maintainer acknowledges receipt and begins working on a fix.
3. A fix is developed and tested before any public disclosure.
4. The reporter is credited in the release notes (unless anonymity is requested).
5. The vulnerability details are published after the fix is released.

## Scope

The following are considered security vulnerabilities:

- Remote code execution
- SQL injection
- Authentication or authorization bypass
- Server-Side Request Forgery (SSRF)
- Path traversal / directory traversal
- Cross-site scripting (XSS) in the web frontend
- Sensitive data exposure (API keys, tokens, user data)
- Denial of service through resource exhaustion

The following are **not** in scope:

- Vulnerabilities in third-party dependencies (report these upstream)
- Issues requiring physical access to the machine
- Social engineering attacks
