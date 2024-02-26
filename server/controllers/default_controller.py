import connexion
import six

from server import util
from auto_migration import *
from server.models.migration_info import MigrationInfo

def auto_migration(vnf):

    auto_migration_destination(vnf)
    return

def test_live_migration_downtime():

    result = test_live_migration()
    return result


def predict_migration_downtime(trained=True):

    result = predict_nn_downtime(trained)
    return result


def do_migration(vnf, node):

    call_migrate(vnf, node)
    return "sucess"


def get_busy_vnfs():

    return get_busy_vnf_info()

    #return ssh_test()
