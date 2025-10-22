
ps -ef | grep oy-gen-pptx | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9

nohup bash -c 'export PYTHONPATH=/opt/app/oy-gen-pptx:$PYTHONPATH ; cd /opt/app/oy-gen-pptx/ppt; /opt/app/oy-gen-pptx/.venv/bin/python start.py' > /tmp/oy-gen-pptx.out 2>&1 &
nohup bash -c 'export PYTHONPATH=/opt/app/oy-gen-pptx:$PYTHONPATH ; cd /opt/app/oy-gen-pptx; /opt/app/oy-gen-pptx/.venv/bin/python ppt_mcp.py' > /tmp/oy-gen-pptx.out 2>&1 &
