resource "oci_core_instance" "generated_oci_core_instance" {
  depends_on = [oci_database_autonomous_database.generated_database_autonomous_database]
  availability_config {
    is_live_migration_preferred = "false"
    recovery_action             = "STOP_INSTANCE"
  }
  availability_domain = var.availability_domain
  compartment_id      = var.compartment_ocid
  create_vnic_details {
    assign_ipv6ip             = "false"
    assign_private_dns_record = "true"
    assign_public_ip          = "true"
    subnet_id                 = var.subnet_ai_subnet_id
  }
  display_name = var.instance_display_name
  instance_options {
    are_legacy_imds_endpoints_disabled = "false"
  }
  metadata = {
    "user_data"           = data.template_cloudinit_config.cloud_init.rendered
    "ssh_authorized_keys" = var.ssh_authorized_keys
  }
  platform_config {
    is_symmetric_multi_threading_enabled = "true"
    type                                 = "AMD_VM"
  }
  shape = var.instance_shape
  shape_config {
    baseline_ocpu_utilization = "BASELINE_1_1"
    memory_in_gbs             = var.instance_flex_shape_memory
    ocpus                     = var.instance_flex_shape_ocpus
  }
  source_details {
    boot_volume_size_in_gbs = var.instance_boot_volume_size
    boot_volume_vpus_per_gb = var.instance_boot_volume_vpus
    source_id               = var.instance_image_source_id
    source_type             = "image"
  }
}
