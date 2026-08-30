# Routing Policies

Implemented policies are `DIGITAL_TEXT_VALIDATE`, `OCR_FAST`,
`OCR_ACCURATE_VISION`, `OCR_ARABIC_COMPLEX`, `OCR_MIXED_LANGUAGE`,
`OCR_RETRY_LOW_QUALITY`, `TEXT_POST_PROCESS`, and `STRUCTURED_EXTRACTION`.

Policies express capability requirements/preferences, local-OCR-first behavior, vision
and structured-output requirements, attempt/failover limits, cost/latency bounds,
provider/model allow/block lists, minimum verification, quality target, fallback,
privacy, and timeout-after-send behavior. Born-digital usable text routes to text
validation; low-quality or complex pages may require verified vision. Provider brands
are not permanent quality tiers.
