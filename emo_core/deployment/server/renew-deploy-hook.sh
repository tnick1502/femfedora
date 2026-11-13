#!/bin/sh
set -eu
/usr/sbin/nginx -t 2>&1
/usr/bin/systemctl reload nginx
