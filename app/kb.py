from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    title: str
    content: str
    keywords: tuple = ()  # extra retrieval hints (synonyms users might use)


KNOWLEDGE_BASE = [
    Section(
        title="Password Reset",
        content=(
            "Users can reset their password from the login page by selecting Forgot Password. "
            "A reset link remains valid for 30 minutes."
        ),
        keywords=("forgot", "login", "sign in", "log in", "locked out", "credentials"),
    ),
    Section(
        title="Two-Factor Authentication",
        content=(
            "Two-factor authentication can be enabled from: "
            "Settings → Security → Two-Factor Authentication. "
            "The application currently supports authenticator applications. "
            "SMS authentication is not supported."
        ),
        keywords=("2fa", "mfa", "authenticator", "sms", "text message", "security", "verification", "otp"),
    ),
    Section(
        title="Subscription Cancellation",
        content=(
            "Users can cancel a subscription from: Settings → Billing → Manage Subscription. "
            "The subscription remains active until the end of the current billing period. "
            "Refunds are reviewed manually by the billing team and are not guaranteed."
        ),
        keywords=("cancel", "billing", "refund", "plan", "unsubscribe", "payment", "charge", "money back"),
    ),
]
