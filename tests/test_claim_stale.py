"""A claim by a seat that differs from the ticket's current holder must
self-heal rather than wedge dispatch.

Found 2026-09-28: workspace-o0n.5.5 had metadata assigned_seat=module-degradation-ts
but was still claimed by subspatial-damage-compartments-reddick, so every claim
attempt failed, tick() kept re-selecting it, and NOTHING else started."""

from harness import beads


def test_claim_releases_a_stale_claim_by_another_seat():
    beads.ensure_initialized()
    proj = beads.create("claim stale proj", "d", issue_type="epic")
    t = beads.create("claim stale ticket", "d", parent=proj["id"])

    beads.assign_to_seat(t["id"], "seat-new")   # earmark while open -> metadata=seat-new
    beads.claim(t["id"], actor="seat-old")      # old holder claims it -> assignee=seat-old

    beads.claim(t["id"], actor="seat-new")      # must self-heal, not raise

    assert beads.show(t["id"]).get("assignee") == "seat-new"
