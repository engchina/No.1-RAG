output "autonomous_data_warehouse_admin_password" {
#   value = random_string.autonomous_data_warehouse_admin_password.result
    value = var.adb_password
}

output "autonomous_data_warehouse_high_connection_string" {
  value = lookup(
    oci_database_autonomous_database.generated_database_autonomous_database.connection_strings[0].all_connection_strings,
    "HIGH",
    "unavailable",
  )
}

output "ssh_to_instance" {
  description = "convenient command to ssh to the instance"
  value       = "ssh -o ServerAliveInterval=10 ubuntu@${oci_core_instance.generated_oci_core_instance.public_ip}"
}

#output "sqlplus_to_oracle" {
#  description = "convenient command to sqlplus to oracle database"
#  value       = "sqlplus vector100/HelloVector__20240101@${oci_core_instance.generated_oci_core_instance.public_ip}:1521/freepdb1"
#}

output "application_url" {
  description = "convenient command to access the application"
  value       = "http://${oci_core_instance.generated_oci_core_instance.public_ip}:8080"
}