"""Read-only iCloud Calendar tool via CalDAV."""

import logging
from datetime import datetime, timedelta, timezone

import caldav

from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

# Set by app.py at startup
caldav_url: str = ""
caldav_username: str = ""
caldav_password: str = ""


def _get_client() -> caldav.DAVClient:
    return caldav.DAVClient(
        url=caldav_url,
        username=caldav_username,
        password=caldav_password,
    )


def _format_event(event: caldav.Event) -> str:
    """Format a single calendar event as readable text."""
    comp = event.icalendar_component
    summary = str(comp.get("SUMMARY", "(no title)"))
    location = comp.get("LOCATION", "")
    description = comp.get("DESCRIPTION", "")

    dtstart = comp.get("DTSTART")
    dtend = comp.get("DTEND")

    start_str = str(dtstart.dt) if dtstart else "?"
    end_str = str(dtend.dt) if dtend else ""

    parts = [f"- {summary}"]
    if end_str:
        parts[0] += f"  ({start_str} → {end_str})"
    else:
        parts[0] += f"  ({start_str})"
    if location:
        parts.append(f"  Location: {location}")
    if description:
        parts.append(f"  Notes: {str(description)[:200]}")
    return "\n".join(parts)


async def read_calendar(
    days: int = 7,
    calendar_name: str = "",
) -> str:
    """Fetch upcoming events from iCloud Calendar."""
    if not caldav_url or not caldav_username or not caldav_password:
        return "Error: CalDAV credentials not configured (CALDAV_URL, CALDAV_USERNAME, CALDAV_PASSWORD)"

    try:
        client = _get_client()
        principal = client.principal()
        calendars = principal.calendars()

        if not calendars:
            return "No calendars found."

        if calendar_name:
            matches = [
                c for c in calendars if calendar_name.lower() in str(c.name).lower()
            ]
            if not matches:
                names = ", ".join(str(c.name) for c in calendars)
                return f"Calendar '{calendar_name}' not found. Available: {names}"
            calendars = matches

        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)

        all_events: list[tuple[datetime, str]] = []
        for cal in calendars:
            cal_name = str(cal.name)
            events = cal.search(start=now, end=end, event=True, expand=True)
            for ev in events:
                comp = ev.icalendar_component
                dtstart = comp.get("DTSTART")
                sort_key = dtstart.dt if dtstart else now
                if hasattr(sort_key, "hour"):
                    if sort_key.tzinfo is None:
                        sort_key = sort_key.replace(tzinfo=timezone.utc)
                else:
                    # All-day event (date, not datetime)
                    sort_key = datetime(
                        sort_key.year, sort_key.month, sort_key.day, tzinfo=timezone.utc
                    )
                formatted = _format_event(ev)
                if len(calendars) > 1:
                    formatted = formatted.replace("- ", f"- [{cal_name}] ", 1)
                all_events.append((sort_key, formatted))

        if not all_events:
            return f"No events in the next {days} days."

        all_events.sort(key=lambda x: x[0])
        lines = [text for _, text in all_events]
        header = f"Events for the next {days} days ({len(all_events)} total):\n"
        return header + "\n".join(lines)

    except Exception as e:
        logger.exception("CalDAV error")
        return f"Calendar error: {e}"


read_calendar_tool = ToolDef(
    name="read_calendar",
    description=(
        "Read upcoming events from the user's iCloud calendar. "
        "Returns event titles, times, locations, and notes. Read-only.\n\n"
        "Known calendars: 'Private' is Jakob's own calendar. 'Freizeit' belongs to his partner."
    ),
    parameters={
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Number of days ahead to look (default 7)",
                "default": 7,
            },
            "calendar_name": {
                "type": "string",
                "description": "Filter to a specific calendar by name (optional, searches all if empty)",
                "default": "",
            },
        },
        "required": [],
    },
    handler=read_calendar,
)
