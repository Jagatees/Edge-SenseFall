# SmartFall Raspberry Pi 5 Setup Guide

This guide explains how to move your SmartFall dashboard project to Raspberry Pi 5 and run email alerts.

## 1. Prepare Raspberry Pi

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip
```

Optional but recommended:

```bash
sudo apt install -y libatlas-base-dev
```

## 2. Get the Project Code

```bash
git clone https://github.com/TenshiRachel/sensefall.git
cd sensefall
git checkout dashboard
```

## 3. Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create `.env` in project root with your values:

```env
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USER=jagateesvaran@gmail.com
ALERT_SMTP_PASS=yjym tvdj cdtt jtqh
ALERT_FROM_EMAIL=jagateesvaran@gmail.com
ALERT_TO_EMAIL=jagateesvaran@gmail.com
EMAIL_COOLDOWN_SEC=60
PORT=5050
OPENCV_AVFOUNDATION_SKIP_AUTH=1
```

Load variables:

```bash
set -a
source .env
set +a
```

## 5. Run Dashboard

```bash
python3 dashboard.py
```

Open in browser:

- Same Pi: `http://127.0.0.1:5050`
- From another device (same Wi-Fi): `http://<PI_IP>:5050`

Get Pi IP:

```bash
hostname -I
```

## 6. Email Alerts on Raspberry Pi

Yes, it works the same on Raspberry Pi.

- SMTP host stays `smtp.gmail.com`
- SMTP port stays `587`
- Same sender/recipient setup in `.env`
- Pi must have internet access

Use dashboard button **Send Test Email** to verify.

## 7. Camera Notes

If camera is not detected, dashboard will show waiting/not connected.

Check camera:

```bash
ls /dev/video*
```

If no device appears, reconnect USB camera and try again.

## 8. Common Issues

### Email not sent: SMTP env vars not fully configured
- `.env` was not loaded before starting app
- Fix: run `set -a; source .env; set +a` then `python3 dashboard.py`

### Gmail auth failure
- App password is wrong/revoked
- 2FA not enabled on Gmail account

### Dashboard not reachable from other devices
- Wrong Pi IP
- Different network/subnet
- Port blocked by firewall/router

## 9. Optional: Auto-start on Boot (systemd)

Create service file:

```bash
sudo nano /etc/systemd/system/smartfall-dashboard.service
```

Paste this (update paths if needed):

```ini
[Unit]
Description=SmartFall Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/sensefall
EnvironmentFile=/home/pi/sensefall/.env
ExecStart=/home/pi/sensefall/.venv/bin/python /home/pi/sensefall/dashboard.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable smartfall-dashboard
sudo systemctl start smartfall-dashboard
sudo systemctl status smartfall-dashboard
```

Logs:

```bash
journalctl -u smartfall-dashboard -f
```
