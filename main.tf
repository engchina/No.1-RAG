provider "oci" {}

resource "oci_core_instance" "generated_oci_core_instance" {
	agent_config {
		is_management_disabled = "false"
		is_monitoring_disabled = "false"
		plugins_config {
			desired_state = "DISABLED"
			name = "Vulnerability Scanning"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Oracle Java Management Service"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "OS Management Service Agent"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "OS Management Hub Agent"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Management Agent"
		}
		plugins_config {
			desired_state = "ENABLED"
			name = "Compute Instance Run Command"
		}
		plugins_config {
			desired_state = "ENABLED"
			name = "Compute Instance Monitoring"
		}
		plugins_config {
			desired_state = "ENABLED"
			name = "Cloud Guard Workload Protection"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Block Volume Management"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Bastion"
		}
	}
	availability_config {
		is_live_migration_preferred = "false"
		recovery_action = "STOP_INSTANCE"
	}
	availability_domain = "bxtG:AP-TOKYO-1-AD-1"
	compartment_id = "ocid1.compartment.oc1..aaaaaaaadsagovrmugvhqvrug74ikajc3xzgyqqbundue4js2qn4g6wlnzmq"
	create_vnic_details {
		assign_ipv6ip = "false"
		assign_private_dns_record = "true"
		assign_public_ip = "true"
		subnet_id = "ocid1.subnet.oc1.ap-tokyo-1.aaaaaaaa5imetr2dm3u5pxdyzk63hku5struudgqxrm5glvgpjfgspa64gba"
	}
	display_name = "aipoc-instance"
	instance_options {
		are_legacy_imds_endpoints_disabled = "false"
	}
	metadata = {
		"user_data" = "IyEvYmluL2Jhc2gKc3VkbyBkZCBpZmxhZz1kaXJlY3QgaWY9L2Rldi9zZGEgb2Y9L2Rldi9udWxsIGNvdW50PTEKZWNobyAiMSIgfCBzdWRvIHRlZSAvc3lzL2NsYXNzL2Jsb2NrL3NkYS9kZXZpY2UvcmVzY2FuCmVjaG8gInkiIHwgc3VkbyAvdXNyL2xpYmV4ZWMvb2NpLWdyb3dmcw=="
		"ssh_authorized_keys" = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDKtwbNDlpP5dLO6rf/KCL8RpiZdX+uKISlnKcuytkE8BhSmeZkRJjUpCM7QatZ6Gbsti7m+SayokY7uK/hx5AruurqYyYtThcZOT9LG9rVG18NGoTfsM7GtJtejQMiQcS+q/Rc3OExHfGn+zlQlJaNYu6H8nU1xxGSfUvew15Vsh6pq1m27XqKcr5YfgMqaVunvsI9V0CSrSDCstFHFVgf7av/5F62GBY6MZPrsk8w8ZEyFalZm+2Yw/DJutPePJOiJeXYJqn/k52FfeBmEIIexHckjTFPdXhEc2OVD95dcK2eGbGy9//yaUE58cVLkMdQboH6svjMG6xbY89N+TTv oracle@k8s-master"
	}
	platform_config {
		is_symmetric_multi_threading_enabled = "true"
		type = "AMD_VM"
	}
	shape = "VM.Standard.E4.Flex"
	shape_config {
		baseline_ocpu_utilization = "BASELINE_1_1"
		memory_in_gbs = "16"
		ocpus = "2"
	}
	source_details {
		boot_volume_size_in_gbs = "100"
		boot_volume_vpus_per_gb = "10"
		source_id = "ocid1.image.oc1.ap-tokyo-1.aaaaaaaakwgkn77bbrpjsioec4xwhciu5wyflm6gyw4rfrn5xm4futo72sha"
		source_type = "image"
	}
}
