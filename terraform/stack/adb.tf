resource "random_string" "autonomous_data_warehouse_admin_password" {
  length      = 16
  min_numeric = 1
  min_lower   = 1
  min_upper   = 1
  min_special = 1
}

resource "oci_database_autonomous_database" "generated_database_autonomous_database" {
#   admin_password                       = random_string.autonomous_data_warehouse_admin_password.result
  admin_password                       = var.adb_password
  autonomous_maintenance_schedule_type = "REGULAR"
  backup_retention_period_in_days      = "1"
  character_set                        = "AL32UTF8"
  compartment_id                       = var.compartment_ocid
  compute_count                        = "2"
  compute_model                        = "ECPU"
  data_storage_size_in_tbs             = "1"
  db_name                              = var.adb_name
  db_version                                     = "23ai"
  db_workload                                    = "DW"
  display_name                                   = var.adb_name
  is_auto_scaling_enabled                        = "false"
  is_auto_scaling_for_storage_enabled            = "false"
  is_dedicated                                   = "false"
  is_mtls_connection_required                    = "true"
  is_preview_version_with_service_terms_accepted = "false"
  license_model                                  = var.license_model
  ncharacter_set                                 = "AL16UTF16"
}

resource "oci_database_autonomous_database_wallet" "generated_autonomous_data_warehouse_wallet" {
  autonomous_database_id = oci_database_autonomous_database.generated_database_autonomous_database.id
#   password               = random_string.autonomous_data_warehouse_admin_password.result
  password               = var.adb_password
  base64_encode_content  = "true"
}

resource "local_file" "generated_autonomous_data_warehouse_wallet_file" {
  content_base64 = oci_database_autonomous_database_wallet.generated_autonomous_data_warehouse_wallet.content
  filename       = "${path.module}/wallet.zip"
}