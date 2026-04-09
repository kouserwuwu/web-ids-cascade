"""列出 Windows 下 Scapy 可见网卡（请在项目 demo 根目录运行：python scripts/tools/list_ifaces.py）。"""
from scapy.arch.windows import get_windows_if_list

ifs = get_windows_if_list()
for i, info in enumerate(ifs):
    # info 里通常有: name (NPF 名), description (描述), ip, mac 等
    print(i)
    print("  name       :", info.get("name"))
    print("  description:", info.get("description"))
    print("  guid       :", info.get("guid"))
    print("  mac        :", info.get("mac"))
    print("  ip         :", info.get("ip"))
    print()
    