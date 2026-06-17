data "oci_core_subnet" "adb_acl_subnet" {
  count     = var.adb_network_access_type == "SECURE_ACCESS_FROM_ALLOWED_IPS_AND_VCNS" && var.adb_acl_notation_type == "VCN" && trimspace(var.adb_acl_subnet_id) != "" ? 1 : 0
  subnet_id = var.adb_acl_subnet_id
}

locals {
  adb_display_name             = trimspace(var.adb_display_name) != "" ? var.adb_display_name : var.adb_name
  adb_private_endpoint_enabled = var.adb_network_access_type == "PRIVATE_ENDPOINT_ONLY" || var.adb_use_private_subnet
  adb_secure_acl_enabled       = var.adb_network_access_type == "SECURE_ACCESS_FROM_ALLOWED_IPS_AND_VCNS"

  adb_acl_cidr_entries = local.adb_secure_acl_enabled && var.adb_acl_notation_type == "CIDR_BLOCK" && trimspace(var.adb_acl_cidr_blocks) != "" ? [
    for cidr in split(",", var.adb_acl_cidr_blocks) : trimspace(cidr)
    if trimspace(cidr) != ""
  ] : []

  adb_acl_vcn_entries = local.adb_secure_acl_enabled && var.adb_acl_notation_type == "VCN" && trimspace(var.adb_acl_vcn_id) != "" ? [
    trimspace(var.adb_acl_subnet_id) != "" ? "${var.adb_acl_vcn_id};${data.oci_core_subnet.adb_acl_subnet[0].cidr_block}" : var.adb_acl_vcn_id
  ] : []

  adb_whitelisted_ips = local.adb_secure_acl_enabled ? concat(local.adb_acl_vcn_entries, local.adb_acl_cidr_entries) : null
}

resource "oci_database_autonomous_database" "generated_database_autonomous_database" {
  admin_password                                 = var.adb_password
  autonomous_maintenance_schedule_type           = "REGULAR"
  backup_retention_period_in_days                = var.adb_backup_retention_period_in_days
  character_set                                  = "AL32UTF8"
  compartment_id                                 = var.compartment_ocid
  compute_count                                  = var.adb_compute_count
  compute_model                                  = var.adb_compute_model
  data_storage_size_in_tbs                       = var.adb_data_storage_size_in_tbs
  db_name                                        = var.adb_name
  db_version                                     = var.adb_db_version
  db_workload                                    = var.adb_workload
  display_name                                   = local.adb_display_name
  is_auto_scaling_enabled                        = var.adb_is_auto_scaling_enabled
  is_auto_scaling_for_storage_enabled            = var.adb_is_auto_scaling_for_storage_enabled
  is_dedicated                                   = "false"
  is_mtls_connection_required                    = var.adb_is_mtls_connection_required
  is_preview_version_with_service_terms_accepted = "false"
  license_model                                  = var.license_model
  ncharacter_set                                 = "AL16UTF16"
  subnet_id                                      = local.adb_private_endpoint_enabled ? var.adb_subnet_id : null
  whitelisted_ips                                = local.adb_secure_acl_enabled ? local.adb_whitelisted_ips : null

  dynamic "resource_pool_summary" {
    for_each = var.adb_is_elastic_pool_enabled ? [1] : []

    content {
      pool_size                = var.adb_resource_pool_size
      pool_storage_size_in_tbs = var.adb_resource_pool_storage_size_in_tbs
    }
  }
}

resource "oci_database_autonomous_database_wallet" "generated_autonomous_data_warehouse_wallet" {
  autonomous_database_id = oci_database_autonomous_database.generated_database_autonomous_database.id
  password               = var.adb_password
  base64_encode_content  = "true"
  generate_type          = "SINGLE"
}

# Save the generated wallet ZIP as a local binary file.
resource "local_file" "wallet_zip" {
  content_base64 = oci_database_autonomous_database_wallet.generated_autonomous_data_warehouse_wallet.content
  filename       = "${path.module}/wallet_full.zip"
}

# Shrink the wallet ZIP before injecting it into cloud-init.
data "external" "wallet_files" {
  depends_on = [local_file.wallet_zip]
  program    = ["bash", "${path.module}/extract_wallet.sh"]
}
