import ni_mon_client, ni_nfvo_client
from ni_mon_client.rest import ApiException
from ni_nfvo_client.rest import ApiException
from create_dashboard import create_dashboard
import datetime
import json
import time
import requests
import paramiko
import numpy as np
import subprocess
import nn as mynn
from config import cfg
from multiprocessing.pool import ThreadPool

# OpenStack Parameters
openstack_network_id = cfg["openstack_network_id"] # Insert OpenStack Network ID to be used for creating SFC
sample_user_data = "#cloud-config\n password: %s\n chpasswd: { expire: False }\n ssh_pwauth: True\n manage_etc_hosts: true\n runcmd:\n - sysctl -w net.ipv4.ip_forward=1"
#ni_nfvo_client_api
ni_nfvo_client_cfg = ni_nfvo_client.Configuration()
ni_nfvo_client_cfg.host=cfg["ni_nfvo"]["host"]
ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))

#ni_monitoring_api
ni_mon_client_cfg = ni_mon_client.Configuration()
ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

#Configuration
NUMBER_TO_MIGRATE = 5
PROFILE_PERIOD = 30
PROFILE_INTERVAL = 1
record = []

#Data
migrating_vnfs = []


def ssh_test():
    print("This is test function")

    ctrl_ssh = get_ssh(cfg["openstack_controller"]["ip"], cfg["openstack_controller"]["username"], cfg["openstack_controller"]["password"])
    ue_ssh = get_ssh(cfg["traffic_controller"]["ip"], cfg["traffic_controller"]["username"], cfg["traffic_controller"]["password"])


    stress_policy = create_stress_policy()[0]


    print("Imposing stress...")
    ssh_command = "sshpass -p %s ssh -o stricthostkeychecking=no %s@%s "
    target_vm = (cfg["migration_test_vm"]["password"], cfg["migration_test_vm"]["username"], "10.10.20.166")
    inner_command = "\"nohup stress-ng --fault {} > /dev/null 2>&1 &\"".format(stress_policy)
    command = (ssh_command + inner_command) % target_vm
    stdin, stdout, stderr = ue_ssh.exec_command(command)


    print("Waitting for release")

    time.sleep(60)

    print("release stress")
    ssh_command = "sshpass -p %s ssh -o stricthostkeychecking=no %s@%s "
    target_vm = (cfg["migration_test_vm"]["password"], cfg["migration_test_vm"]["username"], "10.10.20.166")

    inner_command = "\"pgrep stress-ng | xargs -I{} sudo kill -9 {}\""
    command = (ssh_command + inner_command) % target_vm
    stdin, stdout, stderr = ue_ssh.exec_command(command)


    return 


def check_available_resource(node_id):
    node_info = get_node_info()
    selected_node = [ node for node in node_info if node.id == node_id ][-1]
    flavor = ni_mon_api.get_vnf_flavor(cfg["flavor"]["default"])

    if selected_node.n_cores_free >= flavor.n_cores and selected_node.ram_free_mb >= flavor.ram_mb:
        return (selected_node.n_cores_free - flavor.n_cores)

    return False


def check_active_instance(id):
    status = ni_mon_api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False


def get_node_info():
    query = ni_mon_api.get_nodes()

    response = [ node_info for node_info in query if node_info.type == "compute" and node_info.status == "enabled"]
    response = [ node_info for node_info in response if not (node_info.name).startswith("NI-Compute-82-9")]

    return response


def get_node_ip_from_node_id(node_id):
    node_info = get_node_info()

    for info in node_info:
        if info.id == node_id:
            return info.ip

    print("Cannot get node ip from node id")
    return False


def get_node_id_from_node_ip(node_ip):
    node_info = get_node_info()

    for info in node_info:
        if info.ip == node_ip:
            return info.id

    print("Cannot get node id from node ip")
    return False


def get_vnf_info(vnf_id):
    query = ni_mon_api.get_vnf_instance(vnf_id)
    response = query
    
    return response


def get_vnf_ip_from_vnf_id(vnf_id):
    api_response = ni_mon_api.get_vnf_instance(vnf_id)
    ports = api_response.ports
    network_id = openstack_network_id

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]


def get_node_id_from_vnf_id(vnf_id):

    node_id = get_vnf_info(vnf_id).node_id

    return node_id
    

def get_ssh(ssh_ip, ssh_username, ssh_password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_ip, username=ssh_username, password=ssh_password)
    return ssh


def get_nfvo_vnf_spec():
#    print("5")

    t = ni_nfvo_client.ApiClient(ni_nfvo_client_cfg)

    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(t)
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"]["default"]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec


def set_vnf_spec(vnf_name, node_name):
    vnf_spec = get_nfvo_vnf_spec()
    vnf_spec.vnf_name = vnf_name
    vnf_spec.image_id = cfg["image"]["base"] #client or server
    vnf_spec.node_name = node_name

    return vnf_spec


def deploy_vnf(vnf_spec):
    api_response = ni_nfvo_vnf_api.deploy_vnf(vnf_spec)
    #print(vnf_spec)
    #print(api_response)

    return api_response


def destroy_vnf(id):
    api_response = ni_nfvo_vnf_api.destroy_vnf(id)

    return api_response


# Get instance name.
def get_instance_name(vnf_id):

    node_id = get_vnf_info(vnf_id).node_id
    
    node_ip = get_node_ip_from_node_id(node_id)
    host_ssh = get_ssh(node_ip, cfg["openstack_controller"]["username"], cfg["openstack_controller"]["password"])

    command = "LIBVIRT_DEFAULT_URI=qemu:///system virsh list | grep instance | awk '{ print $1 }'"
    stdin, stdout, stderr = host_ssh.exec_command(command)
    
    dom_id = []
    instance_name = ""
    for line in stdout.readlines():
        dom_id.append(line.rstrip())


    for domain in dom_id:
        command = "LIBVIRT_DEFAULT_URI=qemu:///system virsh dominfo " + domain + " | grep -C 5 " + vnf_id + " | grep 'Name' | awk '{ print $2 }'"
        stdin, stdout, stderr = host_ssh.exec_command(command)

        result = stdout.readlines()
        if len(result) > 0:
            instance_name = result[0].rstrip()
            break

    host_ssh.close()
    return instance_name



def install_target_vnf(node_id):
    spec = set_vnf_spec("Migration-test-VM", node_id)


    instance_id = deploy_vnf(spec)
    limit = 100
    print("VNF id : {}".format(instance_id))
    for i in range(0, limit):
        time.sleep(2)

        # Success to create VNF instance
        if check_active_instance(instance_id):
            break
        elif i == (limit-1):
            destroy_vnf(instance_id)
            print("Failed to deploy VNF")
            return False

    vnf_info=get_vnf_info(instance_id)
    cfg["migration_test_vm"]["ip"] = get_vnf_ip_from_vnf_id(instance_id)
    cfg["migration_test_vm"]["name"] = vnf_info.name
    cfg["migration_test_vm"]["id"] = vnf_info.id
    cfg["migration_test_vm"]["instance_name"] = get_instance_name(instance_id)

    #print(cfg["migration_test_vm"])

    return instance_id



def connect_target_vnf():
    response =""
    try:
        response = get_vnf_info(cfg["migration_test_vm"]["id"])
        response = response.id
        print("Success to connect test_vm from config")
    except:
        print("Try to install new vnf for testing migration")
        if check_available_resource(cfg["shared_host"]["host1"]["name"]):
              response = install_target_vnf(cfg["shared_host"]["host1"]["name"])
        elif check_available_resource(cfg["shared_host"]["host2"]["name"]):
              response = install_target_vnf(cfg["shared_host"]["host2"]["name"])
        else:
            print("No available resource for installing client or server")

    return response



def ssh_keygen(ue_ssh, ip):
    command = "sudo ssh-keygen -f '/home/ubuntu/.ssh/known_hosts' -R %s" % ip
    stdin, stdout, stderr = ue_ssh.exec_command(command, timeout=120)
    #print("ssh-key gen :",stdout.readlines())

    return True



# TODO: improve to generate a stress policy combining dirty page (stackmmap) and page fault.
# Create a set of stress weights applied to the target VM in order of migration round.
def create_stress_policy():
    # if i < 200:
    #     return "sudo ./stress-ng --stackmmap 1"
    # elif i < 400:
    #     return "sudo ./stress-ng --stackmmap 2"
    # elif i < 600:
    #     return "sudo ./stress-ng --stackmmap 3"
    # elif i < 800:
    #     return "sudo ./stress-ng --fault 1"

    # Return random integers from the “discrete uniform” distribution
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint
    policy_arr = np.random.randint(1, 4, NUMBER_TO_MIGRATE)     # converted to stress-ng --fault 1~4 in impose_stress
    return policy_arr


# Inject stress on the target VM.
def impose_stress(ue_ssh, stress_weight):

    ssh_command = "sshpass -p %s ssh -o stricthostkeychecking=no %s@%s "
    target_vm = (cfg["migration_test_vm"]["password"], cfg["migration_test_vm"]["username"], cfg["migration_test_vm"]["ip"])

    inner_command = "\"nohup stress-ng --fault {} > /dev/null 2>&1 &\"".format(stress_weight)
    command = (ssh_command + inner_command) % target_vm
    print("Imposing stress...")
    print(inner_command)
    stdin, stdout, stderr = ue_ssh.exec_command(command)
    return 

    # readlines() reads all lines until EOF (that's why you see the blocking)
    # https://stackoverflow.com/questions/38853031/paramiko-rsync-get-progress-asynchronously-while-command-runs
    # print(stdout.readlines())


# Release stress on the target VM.
def release_stress(ue_ssh):

    ssh_command = "sshpass -p %s ssh -o stricthostkeychecking=no %s@%s "
    target_vm = (cfg["migration_test_vm"]["password"], cfg["migration_test_vm"]["username"], cfg["migration_test_vm"]["ip"])

    inner_command = "\"pgrep stress-ng | xargs -I{} sudo kill -9 {}\""
    command = (ssh_command + inner_command) % target_vm
    stdin, stdout, stderr = ue_ssh.exec_command(command)
    print(stdout)
    return 

# Profile the characteristic of workload on the target VM. It is subjective to the selected stress policy.
def do_profiling(start_time, end_time):
    params = {'start_time': start_time, 'end_time': end_time}

    # target vm id
    target_vm_id = cfg["migration_test_vm"]["id"]

    # get measurements from the migration target vm
    _do_profiling(target_vm_id, params)

    # get measurements from the src host of the vm
    src_host_id = get_node_id_from_vnf_id(target_vm_id)
    print("src_host_id : ", src_host_id)
    _do_profiling(src_host_id, params)

    # get measurements from the dst host of the vm
    #dst_host_ip = cfg["shared_host"]["host2"]["ip"] if get_node_ip_from_node_id(src_host_id) == cfg["shared_host"]["host1"]["ip"] else cfg["shared_host"]["host1"]["ip"]
    #dst_host_id = get_node_id_from_node_ip(dst_host_ip)
    
    #node_name and node_id is same!! so skipped the procedure that changes between ip and id
    dst_host_id = cfg["shared_host"]["host2"]["name"] if src_host_id == cfg["shared_host"]["host1"]["name"] else cfg["shared_host"]["host1"]["name"]
    _do_profiling(dst_host_id, params)
    return


def _do_profiling(host_id, params):

    url = "{}/{}".format(cfg["ni_mon"]["measurement_types_uri"], host_id)
    measurement_types = list(requests.get(url).json())

    #print("What is that : ", measurement_types)
    for measurement_type in measurement_types:
        sum_value = 0.0
        url = "{}/{}/{}".format(cfg["ni_mon"]["measurements_uri"], host_id, measurement_type)
        measurements = requests.get(url, params).json()
        for i in range(len(measurements)):
            value = measurements[i].get("measurement_value")
            sum_value = sum_value + value

        # for each metric, average the values by the profile period
        avg_metric = sum_value / PROFILE_PERIOD
        #print(measurement_type + ": " + str(avg_metric))
        record.append(avg_metric)
    return


# Select a host to which the target VM is migrated.
def select_migration_destination():
    url = "{}/{}".format(cfg["ni_mon"]["vnfs_uri"], cfg["migration_test_vm"]["id"])
    response = requests.get(url).json()
    current_node_id = response.get("node_id")

    if current_node_id == cfg["shared_host"]["host1"]["name"]:
        return cfg["shared_host"]["host2"]["name"]
    else:
        return cfg["shared_host"]["host1"]["name"]




def source_openrc(ctrl_ssh):

    command = "source /opt/stack/devstack/openrc admin demo"
    stdin, stdout, stderr = ctrl_ssh.exec_command(command)

    return


def check_network_topology():

    api_response = ni_mon_api.get_links()
    #print("tt : ", api_response)    
    compute_groups = {}
    for entry in api_response:
        node1_id = entry.node1_id
        node2_id = entry.node2_id

        if "ni-compute" in node2_id :
            if node1_id not in compute_groups:
                compute_groups[node1_id] = []
            compute_groups[node1_id].append(node2_id)

    #'Switch-core-01'
    return compute_groups



def find_related_ni_compute(target_ni_compute, data):
    edge_ni_computes = []

    for switch, ni_compute_list in data.items():
        #print(switch, ni_compute_list)
        if target_ni_compute in ni_compute_list:
            edge_ni_computes.extend([ni for ni in ni_compute_list if ni != target_ni_compute])


    data_center_computes = data.get('Switch-core-01', [])

    return edge_ni_computes, data_center_computes



def auto_migration_destination(vnf_id):

    node_id = get_node_id_from_vnf_id(vnf_id)

    #find the edge
    topology_info = check_network_topology()

    #get node list of edge and remove itself
    edge_computes, data_center_computes = find_related_ni_compute(node_id, topology_info)

    #check resource
    max_available_cores = -1 
    selected_node = None

    for node_id in edge_computes:
        available_cores = check_available_resource(node_id)
    
        if available_cores is not False and available_cores > max_available_cores:
            max_available_cores = available_cores
            selected_node = node_id

    if selected_node is not None:
        call_migrate(vnf_id, selected_node)
        print(f"The selected node with the most available cpu cores: {selected_node}")

    else:
        call_migrate(vnf_id, data_center_computes)
        print("No suitable node found. Migrate to data-center node")

    return selected_node


# Call OpenStack live migration API to migrate the target VM to the selected dst host.
def live_migrate(ctrl_ssh, dst_node):

    global migrating_vnfs

    migrating_vnfs.append(cfg["migration_test_vm"]["name"])
    # command = "source /opt/stack/devstack/openrc admin demo && openstack server migrate --live %s --block-migration %s" \
    #           % (dst_node, cfg["test_vm"]["os_name"])
    # command = "source /opt/stack/devstack/openrc admin demo && openstack server migrate --live-migration --shared-migration"\
    #           " --os-compute-api-version 2.88 --host %s %s" \
    #           % (dst_node, cfg["test_vm"]["os_name"])
    command = "source /opt/stack/devstack/openrc admin demo && nova live-migration {} {}".format(cfg["migration_test_vm"]["name"], dst_node)

    print(command)
    # FIXME: check timeout is needed indeed
    stdin, stdout, stderr = ctrl_ssh.exec_command(command, timeout=120)
    print("migrate readout :",stdout.readlines())

    migrating_vnfs.remove(cfg["migration_test_vm"]["name"])
    return


# Call OpenStack live migration API to migrate the target VM to the selected dst host.
def call_migrate(vnf_id, dst_node_id):

    global migrating_vnfs
    print("Migration is started")
    migrating_vnfs.append(vnf_id)
    vnf_name = get_vnf_info(vnf_id).name
    ctrl_ssh = get_ssh(cfg["openstack_controller"]["ip"], cfg["openstack_controller"]["username"],
                       cfg["openstack_controller"]["password"])

    source_openrc(ctrl_ssh)
    command = "source /opt/stack/devstack/openrc admin demo && openstack server migrate --os-compute-api-version 2.56 --host {} {}".format(dst_node_id, vnf_name)
    stdin, stdout, stderr = ctrl_ssh.exec_command(command, timeout=60)

    while(True):
        #print("waiting for verifying resize in migration")
        time.sleep(5)
        if get_vnf_info(vnf_id).status == "VERIFY_RESIZE" :
            command = "source /opt/stack/devstack/openrc admin demo && openstack server migrate confirm {}".format(vnf_name)
            stdin, stdout, stderr = ctrl_ssh.exec_command(command, timeout=60)
            break

    migrating_vnfs.remove(vnf_id)
    ctrl_ssh.close()

    return



# Compute total migration time (MT) in second from OpenStack live migration API.
def calculate_migration_time():
    # TODO: for now token and microversion are reset every migration round
    # get an API token
    headers = {"Content-Type": "application/json"}
    data = json.load(open("openstack-credentials.json", 'r'))
    url = "{}".format(cfg["openstack_controller"]["auth_api_uri"])
    response = requests.post(url, headers=headers, json=data)
    print(response)
    token = response.headers["x-subject-token"]
    print(token)

    # get the latest Microversion supported
    headers = {"X-Auth-Token": token}
    url = "{}".format(cfg["openstack_controller"]["compute_api_uri"])
    response = requests.get(url, headers=headers).json()
    micro_version = "compute " + response.get("version").get("version")
    print(micro_version)

    # loop until migration completed
    while True:
        time.sleep(10)   # checking interval
        try:
            # request
            headers = {"X-Auth-Token": token, "OpenStack-API-Version": micro_version}
            url = "{}/{}".format(cfg["openstack_controller"]["compute_api_uri"], "os-migrations?limit=1")
            response = requests.get(url, headers=headers, timeout=5).json()
            latest_migration = response.get("migrations")[0]
            #print(latest_migration)

            if latest_migration.get("instance_uuid") != cfg["migration_test_vm"]["id"]:
                # the latest migration decision must be made for the target epc
                print("ERROR:the latest migration decision must be made for the target epc")
                exit(1)

            status = latest_migration.get("status")
            if status != "running":
                if status == "completed":
                    #time_done = datetime.datetime.fromisoformat(latest_migration.get("updated_at"))
                    #time_started = datetime.datetime.fromisoformat(latest_migration.get("created_at"))
                    time_done = datetime.datetime.strptime(latest_migration.get("updated_at"),"%Y-%m-%dT%H:%M:%S.%f")
                    time_started = datetime.datetime.strptime(latest_migration.get("created_at"),"%Y-%m-%dT%H:%M:%S.%f")

                    total_migration_time = (time_done - time_started).seconds
                    print("total_mt_time : ",total_migration_time)
                    record.append(total_migration_time)
                    return True
                elif status == "error":
                    # TODO: handle status of "error". must go to next round
                    return False
                elif status == "failed":
                    return False
                else:
                    #print(status)
                    time.sleep(1)   # print buffer
                    continue
            print("Migration is ongoing...check again in 5s")

        # https://calssess.tistory.com/92
        except requests.exceptions.Timeout as e:
            print(e)
            continue
    return




# Make the client VM send ping requests to the target VM using nping command.
def send_ue_nping(ue_ssh, i):
    command = "sudo nping --icmp --count 0 --delay {}s {} > {}.round{}"\
        .format(cfg["traffic_controller"]["ping_interval"], cfg["migration_test_vm"]["ip"], cfg["traffic_controller"]["nping_file"], i)
    stdin, stdout, stderr = ue_ssh.exec_command(command)

    print("send ue nping")
    return


# TODO: remove the nping file to avoid possible file creation error in UE
# Make the client VM stop sending ping requests to the target VM using nping command.
def stop_ue_nping(ue_ssh):
    # send SIGINT (ctrl + c) to get the statistics nping provides
    # https://stackoverflow.com/questions/5789642/how-to-send-controlc-from-a-bash-script
    command = "pgrep nping | xargs -I{} sudo kill -INT {}"
    stdin, stdout, stderr = ue_ssh.exec_command(command)
    # print(stdout.readlines())
    return

# Compute total ping loss count using nping command.
# You can derive ping service downtime (DT) from ping_loss_cnt * ping_interval
def calculate_service_downtime_nping(ue_ssh, i):
    command = "tail -n2 nping.txt.round{}".format(i) + "| grep Lost | awk '{print $12}'"
    stdin, stdout, stderr = ue_ssh.exec_command(command)
    loss_pkt_count = int(stdout.readlines()[0])
    print("loss_pkt_count : ",loss_pkt_count)
    record.append(loss_pkt_count)
    return


# Get VM (freeze) downtime time in ms from dst host KVM.
def calculate_vm_downtime(dst_node):

    vm_downtime = 0
    node_ip = get_node_ip_from_node_id(dst_node)
    dst_host_ssh = get_ssh(node_ip, cfg["openstack_controller"]["username"], cfg["openstack_controller"]["password"])

    # TODO: the command assumes there is only one VM running in the dst host. meaning, do not migrate VMs simultaneously.
    # TODO: find a better way to execute the commands
    # Ensure that the prefix is needed to execute the virsh command in remote
    # https://askubuntu.com/questions/1066230/cannot-execute-virsh-command-through-ssh-on-ubuntu-18-04
    # command = "LIBVIRT_DEFAULT_URI=qemu:///system virsh list | sed -n '3p' | awk '{print $1}' | xargs -I{} LIBVIRT_DEFAULT_URI=qemu:///system virsh domjobinfo {} --completed --keep-completed | grep downtime | awk '{print $3}'"
    command = "LIBVIRT_DEFAULT_URI=qemu:///system virsh list | grep " + cfg["migration_test_vm"]["instance_name"]+ " | awk '{print $1}'"
    stdin, stdout, stderr = dst_host_ssh.exec_command(command)
    dom_id = str(stdout.readlines()[0]).rstrip()
    print("dom_id : ",dom_id)

    command = "LIBVIRT_DEFAULT_URI=qemu:///system virsh domjobinfo " + dom_id + " --completed --keep-completed | grep downtime | awk '{print $3}'"
    stdin, stdout, stderr = dst_host_ssh.exec_command(command)
    try:
        vm_downtime = int(stdout.readlines()[0])
        print("vm_downtime : ",vm_downtime)
        record.append(vm_downtime)
    except:
        print("Failed to get KVM data")
    dst_host_ssh.close()
    return vm_downtime

def test_live_migration():

    f = open("test_monitor.csv", "a") #for generate validation data set
    # https://stackoverflow.com/a/14299004/5204099
    # match the number of processes needed for profiling below
    # pool = ThreadPool(processes=1)

    print("Starting auto_migration.py")
    #Connect target VNF or deploy new VNF for migration testing
    vnf_id = connect_target_vnf()

    ML_mydashboard_url = create_dashboard(vnf_instances=[[ni_mon_api.get_vnf_instance(vnf_id)]],dashboard_name="Migration")

    ctrl_ssh = get_ssh(cfg["openstack_controller"]["ip"], cfg["openstack_controller"]["username"], cfg["openstack_controller"]["password"])
    ue_ssh = get_ssh(cfg["traffic_controller"]["ip"], cfg["traffic_controller"]["username"], cfg["traffic_controller"]["password"])

    print("Finishing ssh connection setup")

    time.sleep(60) #for taking screenshot


    source_openrc(ctrl_ssh)
    stress_policy = create_stress_policy()
    

    total_vm_downtime = 0

    
    for i in range(NUMBER_TO_MIGRATE):
        print("\033[31m"+"Migration Round [%d] \033[0m" % i)

        # Impose stress on the migration target vm
        # The stress represents the workload's characteristics: read-intensive, write-intensive or
        release_stress(ue_ssh)  # for safety
        impose_stress(ue_ssh, stress_policy[i])

        # Start Profile Phase for vm live migration
        # Profile the vm's characteristics from measurements for the PROFILE_PERIOD that proceeds the actual migration
        # Make sure collectd interval (1s) == PROFILE_INTERVAL (1s)
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=PROFILE_PERIOD)

        if str(start_time)[-1]!='Z':
            start_time = str(start_time.isoformat()) + 'Z'

        if str(end_time)[-1]!='Z':
            end_time = str(end_time.isoformat())+ 'Z'

        time.sleep(PROFILE_PERIOD + 3)      # buffer to avoid stress in cold state


        do_profiling(start_time, end_time)
        

        # Start Migration Phase
        dst_node = select_migration_destination()
        print("dst_host_id : ", dst_node)
        live_migrate(ctrl_ssh, dst_node)

        # Make UE start sending ping to the target VM

        #stop_ue_nping(ue_ssh)   # for safety
        #send_ue_nping(ue_ssh, i)

        # Check migration status.
        # if the migration is ongoing, then block until it is completed.
        # else, proceed to get the total migration time (MT).



        migrated = calculate_migration_time()
        # Make ue stop sending ping because highly expected that service downtime already occurred (if migration done)
        # stop_ue_ping(ue_ssh)
        # stop_ue_nping(ue_ssh)

        #print("migration: {}".format(migrated))
        if migrated is True:

            # calculate_service_downtime(ue_ssh, i)
            #calculate_service_downtime_nping(ue_ssh, i)

            # Get VM (freeze) downtime from dst host QEMU.
            total_vm_downtime = total_vm_downtime + calculate_vm_downtime(dst_node)
            # Compute ping service downtime (indeed packet lost count).

            # Write the record of the current migration to the dataset file
            # TODO: write column names in header
            f.writelines(str(record).lstrip('[').rstrip(']') + '\n')
            #print(record)

        # Release the stress imposed on the vm
        #release_stress(ue_ssh)

        # Initialize the record
        record.clear()

        # Take a rest until all the system states get stable to be ready for the next migration
        time.sleep(20)
        print("============================================================") 

    
    
    ctrl_ssh.close()
    ue_ssh.close()
    f.close()

    destroy_vnf(vnf_id) 
    
    return "Total {} migration, average vm_downtime : {} Grafana dashboard : {}".format(i+1, float(total_vm_downtime/(i+1) ),ML_mydashboard_url)


def predict_nn_downtime(trained):

    response=mynn.ml(trained)

    print("response : ", response)
    return "Accuracy : {}".format(response)


def get_busy_vnf_info():

    global migrating_vnfs

    #domjobinfo from all-compute node or otherway to get busy instance
   
    #make list busy_from_backend
    busy_from_backend = []
    
    migrating_vnfs = migrating_vnfs + busy_from_backend
    return migrating_vnfs




