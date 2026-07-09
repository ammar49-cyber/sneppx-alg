# Incident Response Framework

SneppX-ALG provides a comprehensive incident response framework aligned with NIST SP 800-61 Rev 2.

## Phases

1. **Preparation** - Tools, playbooks, training, communication plans
2. **Detection & Analysis** - Alert triage, initial scope, evidence collection
3. **Containment, Eradication & Recovery** - Short-term and long-term actions
4. **Post-Incident Activity** - Lessons learned, evidence retention, metrics

## Playbook System

The playbook engine supports:
- Automated and manual steps
- Conditions, variables, and branching
- Sub-playbook composition
- Audit logging of all actions
- Rollback procedures

## Evidence Management

The evidence manager provides:
- Cryptographic chain-of-custody (hash chains)
- Encryption and compression at rest
- Export/import for legal proceedings
- Full-text search across evidence
- Tamper detection

## Forensics Modules

- Memory analysis (Volatility-compatible plugin system)
- Disk image analysis
- Network capture (PCAP/NETFLOW) analysis
- Registry hive analysis
- Malware scanning with YARA
- Timeline/Super-timeline generation

## Reporting

All IR activities generate:
- Executive summaries
- Technical findings
- Remediation recommendations
- Compliance mappings (NIST, ISO 27001, PCI DSS)
