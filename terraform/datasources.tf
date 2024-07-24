data "template_file" "cloud_init_file" {
  template = file("./cloud_init/bootstrap.template.yaml")

  vars = {
    oci_database_autonomous_database_connection_string = base64gzip("admin/${random_string.autonomous_data_warehouse_admin_password.result}@${lookup(oci_database_autonomous_database.generated_database_autonomous_database.connection_strings[0].all_connection_strings,"HIGH","unavailable")}")
    oci_database_autonomous_database_wallet_content    = oci_database_autonomous_database_wallet.generated_autonomous_data_warehouse_wallet.content
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