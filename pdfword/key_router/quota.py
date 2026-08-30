from __future__ import annotations

from .exceptions import QuotaExceeded
from .models import ProviderAccount, RequestContext


def quota_limits_for(
    account: ProviderAccount, context: RequestContext
) -> tuple[int, int, float]:
    request_limit = max(0, int(account.daily_request_limit))
    token_limit = max(0, int(account.daily_token_limit))
    cost_limit = max(0.0, float(account.daily_cost_limit))
    estimated_cost = context.estimated_cost
    if estimated_cost is None:
        raise QuotaExceeded("price_unknown")
    if float(estimated_cost) > 0 and cost_limit <= 0:
        raise QuotaExceeded("paid_without_budget")
    return request_limit, token_limit, cost_limit
