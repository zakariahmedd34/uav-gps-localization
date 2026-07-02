# Test Flight — Data to Send Me

Hi team — I'm not on-site today. To test our localization pipeline I need **3 things** from
this flight. Please get all three.

**1. GoPro video**
Start recording before takeoff, keep it running the whole flight (one continuous file).
Settings stay as I set them yesterday (4K, Linear, stabilization off, GPS on).

**2. The Pixhawk log — the `.bin` file**
The Pixhawk (the flight controller) saves a `.bin` log to its own microSD card. That one file
has the GPS, altitude, and attitude we need.
- Easiest: pull the **microSD card** from the Pixhawk and copy the `.bin` (in the `LOGS/` folder), or
- Mission Planner → **DataFlash Logs → Download Via Mavlink** (over USB).
- Please send the **`.bin`**, not the `.tlog`/`.rlog`.

**3. The flags' real GPS coordinates**
Note the exact GPS location of each flag you place on the ground (phone GPS is fine). Label them
Flag 1, Flag 2… This is what lets me check how accurate our results are.

**One small thing during the flight:** right after takeoff, do a quick **wing-rock** (sharp
left-right roll) — it helps me line up the video with the log.

Thanks! Video + `.bin` + flag coordinates = I can run and test the whole pipeline.
