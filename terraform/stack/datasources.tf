data "template_file" "cloud_init_file" {
  template = file("./cloud_init/bootstrap.template.yaml")

  vars = {
    oci_database_autonomous_database_connection_string = base64gzip("admin/${var.adb_password}@${lower(var.adb_name)}_high")
    oci_database_autonomous_database_wallet_content    = data.external.wallet_files.result.wallet_content
    output_compartment_ocid                            = var.compartment_ocid
    adb_name                                           = var.adb_name
    adb_ocid                                           = oci_database_autonomous_database.generated_database_autonomous_database.id
    adb_password                                       = var.adb_password
    adb_dsn                                            = "${lower(var.adb_name)}_high"
    application_port                                   = tostring(var.application_port)
  }
}


data "template_cloudinit_config" "cloud_init" {
  gzip          = true
  base64_encode = true

  part {
    filename     = "bootstrap.yaml"
    content_type = "text/cloud-config"
    content      = data.template_file.cloud_init_file.rendered
  }
}
