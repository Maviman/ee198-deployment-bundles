"""PC-side controller for the portable N=1 deployment bundle.

Pipeline per control tick (10 Hz):
    poses in -> ObservationAdapter (vendored, exact training contract)
             -> OnnxPolicy (deterministic mean action, no torch required)
             -> EspLink (UDP command packet to the car's ESP32)
"""
