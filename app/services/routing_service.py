"""
Routing service.

Separates "what team was this routed to" (decided inside the LangGraph
workflow) from "what happens as a result" (notifying that team). Keeping
this as its own thin service means the notification channel could change
(SMTP today, maybe Slack later) without touching the graph or database
code at all.

Notification failures are logged, not raised — a failed notification
email should not retroactively undo a successful classification/routing
decision that's already been persisted.
"""

import logging

from app.database.models import Email, RoutingResult
from app.integrations.smtp_client import SMTPSendError, send_routing_notification

logger = logging.getLogger(__name__)

# In a real deployment this would be a lookup table (team_id -> team inbox
# address), likely itself stored in the database or the org knowledge base.
# Hardcoded here since Phase 9's scope is demonstrating the SMTP mechanism,
# not building a full team-directory feature.
TEAM_NOTIFICATION_ADDRESS_TEMPLATE = "team-{team_id}@example.com"


def notify_destination_team(email: Email, routing_result: RoutingResult) -> None:
    to_address = TEAM_NOTIFICATION_ADDRESS_TEMPLATE.format(team_id=routing_result.team_id)
    try:
        send_routing_notification(
            to_address=to_address,
            original_sender=email.sender,
            subject=email.subject,
            summary=routing_result.summary,
            team_name=routing_result.team,
        )
    except SMTPSendError as exc:
        logger.error("Failed to notify team %s for email %s: %s", routing_result.team, email.id, exc)
