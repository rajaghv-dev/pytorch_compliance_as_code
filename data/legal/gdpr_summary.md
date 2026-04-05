# GDPR (Regulation 2016/679) - Key Provisions Summary

## Raw Legal Text Files (Chapter-wise)

| Chapter | File Path | Content |
|---------|-----------|---------|
| 1. General Provisions | `data/legal/gdpr/chapters/01_general_provisions.md` | Articles 1-4 |
| 2. Principles | `data/legal/gdpr/chapters/02_principles.md` | Articles 5-11 |
| 3. Data Subject Rights | `data/legal/gdpr/chapters/03_data_subject_rights.md` | Articles 12-23 |
| 4. Controller/Processor | `data/legal/gdpr/chapters/04_controller_processor.md` | Articles 24-39 |
| 5. Transfers | `data/legal/gdpr/chapters/05_transfers.md` | Articles 44-50 |

---

## Overview
- **Title**: Regulation (EU) 2016/679 of the European Parliament and of the Council
- **Date**: 27 April 2016
- **Effective Date**: 25 May 2018
- **Purpose**: Protection of natural persons with regard to the processing of personal data

---

## Key Articles for AI/ML Systems

### Article 5 - Principles
1. Lawfulness, fairness, transparency
2. Purpose limitation
3. Data minimisation
4. Accuracy
5. Storage limitation
6. Integrity and confidentiality
7. Accountability

### Article 6 - Lawful Processing
- Consent
- Contract performance
- Legal obligation
- Vital interests
- Public task
- Legitimate interests

### Article 7 - Conditions for Consent
- Clear affirmative action
- Freely given, specific, informed, unambiguous
- Easy to withdraw

### Article 9 - Special Categories
- Biometric data
- Health data
- Political opinions
- Religious beliefs
- **Requires explicit consent**

### Article 13 & 14 - Information to Provide
- Controller identity
- Purpose of processing
- Data categories
- Recipients
- Retention period
- Data subject rights

### Article 15 - Right of Access
- Confirmation of processing
- Copy of personal data
- Purpose of processing
- Categories of data
- Recipients
- Retention period

### Article 16 - Right to Rectification
- Rectify inaccurate data
- Complete incomplete data

### Article 17 - Right to Erasure ('Right to be Forgotten')
- Withdraw consent
- Data no longer necessary
- Object to processing
- Unlawful processing
- Erasure for compliance

### Article 18 - Right to Restriction
- Accuracy contested
- Processing unlawful
- Controller no longer needs data
- Pending verification

### Article 20 - Right to Data Portability
- Receive personal data in structured format
- Transfer to another controller

### Article 21 - Right to Object
- Object to processing based on legitimate interests
- Object to direct marketing

### Article 22 - Automated Decision-Making
- **Right not to be subject to decisions based solely on automated processing**
- Includes profiling
- Rights to human intervention
- Right to challenge decisions

### Article 25 - Data Protection by Design
- Pseudonymisation
- Data minimisation
- Encryption
- Technical safeguards

### Article 30 - Records of Processing
- Controller and processor records
- Processing purposes
- Data categories
- Recipients
- Transfers
- Retention periods
- Security measures

### Article 32 - Security of Processing
- Pseudonymisation and encryption
- Confidentiality, integrity
- Resilience of systems
- Timely restoration
- Regular testing

### Article 33 - Breach Notification
- Notify supervisory authority within 72 hours
- Document the breach

### Article 34 - Communication to Data Subject
- Notify affected individuals
- Describe nature of breach

### Article 35 - Data Protection Impact Assessment
- **Required for high-risk processing**
- Systematic evaluation
- Large-scale processing
- Profiling
- Predictive analytics

### Article 44 - General Principle for Transfers
- Adequacy decision
- Appropriate safeguards
- Binding corporate rules

---

## Key Obligations for AI/ML

### Data Minimisation (Art. 5(1)(c))
Only collect data necessary for the specific purpose.

### Purpose Limitation (Art. 5(1)(b))
Data collected for one purpose cannot be used for another without new consent.

### Right to Explanation (Art. 22)
- Automated decisions require human oversight
- Data subjects can challenge decisions

### Data Protection by Design (Art. 25)
- Build privacy into AI systems
- Pseudonymisation
- Encryption

### DPIA (Art. 35)
Required for:
- Large-scale profiling
- Predictive algorithms
- Biometric processing
- Health data processing

### Security (Art. 32)
- Encryption at rest and in transit
- Access controls
- Audit logs

### Breach Response (Art. 33-34)
- 72-hour notification to authority
- Affected individuals notified if high risk

---

## Mapping to PyTorch/MLOps

| GDPR Article | PyTorch Component | Obligation |
|-------------|-------------------|------------|
| Art. 5, 25 | DataLoader, Dataset | Data minimisation, privacy by design |
| Art. 9 | Biometric features | Explicit consent, special category handling |
| Art. 15-22 | Model inference | Right to explanation, portability |
| Art. 17 | Model deletion | Right to erasure |
| Art. 22 | Decision models | Human oversight, no fully automated decisions |
| Art. 30 | Training pipeline | Record-keeping |
| Art. 32 | Model security | Encryption, access control |
| Art. 33-34 | Deployed models | Breach notification |
| Art. 35 | Training pipelines | DPIA for large-scale processing |

---

## Penalties
- Up to EUR 20 million or 4% of global annual turnover
- Whichever is higher

---

*Source: EUR-Lex - https://eur-lex.europa.eu/eli/reg/2016/679/oj*