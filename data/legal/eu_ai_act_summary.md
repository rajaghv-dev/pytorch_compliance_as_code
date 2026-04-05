# EU AI Act - Key Articles for LLM Processing

## Raw Legal Text Files (Chapter-wise)

| Chapter | File Path | Content |
|---------|-----------|---------|
| 1. Preamble | `data/legal/eu_ai_act/chapters/01_preamble.md` | Recitals and Whereas clauses |
| 2. General Provisions | `data/legal/eu_ai_act/chapters/02_general_provisions.md` | Articles 1-4 |
| 3. Prohibited Practices | `data/legal/eu_ai_act/chapters/03_prohibited_practices.md` | Articles 5-7 |
| 4. High-Risk Systems | `data/legal/eu_ai_act/chapters/04_high_risk_systems.md` | Articles 8-20 |
| 5. Obligations | `data/legal/eu_ai_act/chapters/05_obligations.md` | Articles 21-28 |
| 6. Transparency | `data/legal/eu_ai_act/chapters/06_transparency.md` | Article 50 |
| 7. Innovation | `data/legal/eu_ai_act/chapters/07_innovation.md` | Articles 56-58 |

---

## Quick Reference: Articles with Obligations

### Prohibited AI Practices (Article 5)
| Article | Requirement | Obligation Type |
|---------|-------------|-----------------|
| 5(1)(a) | Subliminal techniques to distort behavior | Complete prohibition |
| 5(1)(b) | Exploiting weaknesses in persons | Complete prohibition |
| 5(1)(c) | Social scoring by public authorities | Complete prohibition |
| 5(1)(d) | Real-time remote biometric identification in public spaces | Prohibited with exceptions |

### High-Risk Requirements (Articles 9-20)
| Article | Requirement | Applies To |
|---------|-------------|------------|
| 9 | Risk management system | Provider |
| 10 | Data governance and quality | Provider |
| 11 | Technical documentation | Provider |
| 12 | Record-keeping | Provider |
| 13 | Transparency | Provider |
| 14 | Human oversight | Provider |
| 15 | Accuracy, robustness, cybersecurity | Provider |
| 17 | Quality management system | Provider |
| 18 | Post-market monitoring | Provider |
| 19 | Reporting serious incidents | Provider |
| 20 | Logging obligations | Provider |

### Deployer Obligations (Articles 22-23)
| Article | Requirement |
|---------|-------------|
| 22(1) | Use in accordance with instructions |
| 22(2) | Assign human overseers |
| 22(3) | Monitor operation |

### Transparency (Article 50)
| Section | Requirement |
|---------|-------------|
| 50(1) | Inform users of AI interaction |
| 50(2) | Mark AI-generated content |

---

## Article-by-Article Summary

### Article 5: Prohibited Practices
- Subliminal manipulation of behavior
- Exploitation of vulnerabilities  
- Social scoring by public authorities
- Real-time biometric identification (with exceptions in Article 6)

### Article 9: High-Risk Requirements
1. Risk management system
2. Data governance
3. Technical documentation
4. Record-keeping
5. Transparency
6. Human oversight
7. Accuracy, robustness, cybersecurity

### Article 10: Risk Management
- Continuous process throughout lifecycle
- Identify and mitigate risks
- Document risk management activities

### Article 11: Data Governance
- Relevant, representative, error-free data
- Appropriate data governance practices
- Training/validation/testing data quality

### Article 12: Technical Documentation
- Required before market placement
- Information for conformity assessment
- Must be kept up-to-date

### Article 13: Record-Keeping
- Automatic logging of events
- Traceability of outputs
- Enable post-market monitoring

### Article 14: Transparency
- Enable deployers to interpret outputs
- Provide instructions for use
- Make capabilities/limitations clear

### Article 15: Human Oversight
- Natural person oversight capability
- Prevent or minimize risks
- Ability to understand, intervene, direct

### Article 16: Accuracy & Security
- Appropriate accuracy level
- Handle edge cases gracefully
- Resilience to attacks

### Article 21: Provider Obligations
- Ensure compliance with Title II requirements
- Draw up technical documentation
- Conduct conformity assessment
- Affix CE marking
- Draw up EU declaration of conformity

### Article 22: Deployer Obligations
- Use according to instructions
- Assign human overseers
- Monitor operation
- Report incidents

### Article 50: Transparency Obligations
- Inform users of AI interaction
- Mark AI-generated content (audio/image/video/text)
- Disclose when content is artificially generated

### Article 56: Regulatory Sandboxes
- Establish AI regulatory sandboxes
- Enable testing under controlled conditions
- Support innovation while ensuring compliance

---

## MLOps Stage Mapping

| MLOps Stage | Applicable AI Act Articles |
|-------------|---------------------------|
| Data Collection | Article 11 (Data Governance) |
| Model Training | Article 9, 10 (Risk Management) |
| Model Validation | Article 9, 17 (Quality Management) |
| Model Deployment | Article 14, 15 (Transparency, Oversight) |
| Inference/Monitoring | Article 18, 19 (Post-market, Reporting) |
| Model Retiring | Article 5, GDPR Art. 17 (Deletion) |

---

## Compliance Check: PyTorch APIs

### torch.save() / torch.load()
- **Relevant Articles**: 9, 13, 14, 19, 20
- **Obligations**: 
  - Record-keeping (logging what was saved)
  - Technical documentation
  - Post-market monitoring for serialized models
  - Transparency about model provenance

### torch.nn.Module
- **Relevant Articles**: 9, 10, 11, 14, 15, 17
- **Obligations**:
  - Risk management system
  - Data governance (if training)
  - Human oversight capabilities
  - Accuracy/robustness requirements

### torch.optim.SGD
- **Relevant Articles**: 10, 11
- **Obligations**:
  - Risk management for optimization process
  - Documentation of training procedures

### torch.utils.data.DataLoader
- **Relevant Articles**: 11, 14
- **Obligations**:
  - Data governance
  - Transparency about data processing