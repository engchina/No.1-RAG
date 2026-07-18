variable "availability_domain" {
  default = "bxtG:AP-TOKYO-1-AD-1"
}

variable "compartment_ocid" {
  default = ""
}

variable "vcn_ai_vcn_id" {
  description = "VCN OCID used by the Resource Manager form for subnet filtering"
  type        = string
  default     = ""
}

variable "adb_display_name" {
  description = "Autonomous Database display name. Leave blank to use adb_name."
  type        = string
  default     = ""
}

variable "adb_name" {
  description = "Autonomous Database database name"
  type        = string
  default     = "AIPOCADB"
}

variable "adb_password" {
  type      = string
  sensitive = true
  default   = ""
}

variable "adb_workload" {
  description = "Autonomous Database workload type"
  type        = string
  default     = "DW"

  validation {
    condition     = contains(["OLTP", "DW", "AJD", "APEX", "LH"], var.adb_workload)
    error_message = "adb_workload must be one of OLTP, DW, AJD, APEX, or LH."
  }
}

variable "adb_db_version" {
  description = "Autonomous Database version"
  type        = string
  default     = "26ai"

  validation {
    condition     = contains(["19c", "23ai", "26ai"], var.adb_db_version)
    error_message = "adb_db_version must be one of 19c, 23ai, or 26ai."
  }
}

variable "adb_compute_model" {
  description = "Autonomous Database compute model"
  type        = string
  default     = "ECPU"

  validation {
    condition     = contains(["ECPU", "OCPU"], var.adb_compute_model)
    error_message = "adb_compute_model must be ECPU or OCPU."
  }
}

variable "adb_compute_count" {
  description = "Autonomous Database compute count"
  type        = number
  default     = 2

  validation {
    condition     = var.adb_compute_count > 0
    error_message = "adb_compute_count must be greater than 0."
  }
}

variable "adb_is_auto_scaling_enabled" {
  description = "Enable Autonomous Database compute auto scaling"
  type        = bool
  default     = false
}

variable "adb_data_storage_size_in_tbs" {
  description = "Autonomous Database storage size in TB"
  type        = number
  default     = 1

  validation {
    condition     = var.adb_data_storage_size_in_tbs > 0
    error_message = "adb_data_storage_size_in_tbs must be greater than 0."
  }
}

variable "adb_is_auto_scaling_for_storage_enabled" {
  description = "Enable Autonomous Database storage auto scaling"
  type        = bool
  default     = false
}

variable "adb_is_elastic_pool_enabled" {
  description = "Enable Autonomous Database elastic pool configuration"
  type        = bool
  default     = false
}

variable "adb_resource_pool_size" {
  description = "Autonomous Database elastic pool size. Used when adb_is_elastic_pool_enabled is true."
  type        = number
  default     = 0

  validation {
    condition     = var.adb_resource_pool_size >= 0
    error_message = "adb_resource_pool_size must be 0 or greater."
  }
}

variable "adb_resource_pool_storage_size_in_tbs" {
  description = "Autonomous Database elastic pool storage size in TB. Used when adb_is_elastic_pool_enabled is true."
  type        = number
  default     = 0

  validation {
    condition     = var.adb_resource_pool_storage_size_in_tbs >= 0
    error_message = "adb_resource_pool_storage_size_in_tbs must be 0 or greater."
  }
}

variable "license_model" {
  type    = string
  default = ""
}

variable "adb_backup_retention_period_in_days" {
  description = "Autonomous Database automatic backup retention period in days"
  type        = number
  default     = 1

  validation {
    condition     = var.adb_backup_retention_period_in_days >= 1
    error_message = "adb_backup_retention_period_in_days must be 1 or greater."
  }
}

variable "adb_network_access_type" {
  description = "Autonomous Database network access mode"
  type        = string
  default     = "PRIVATE_ENDPOINT_ONLY"

  validation {
    condition = contains([
      "PUBLIC_ENDPOINT",
      "SECURE_ACCESS_FROM_ALLOWED_IPS_AND_VCNS",
      "PRIVATE_ENDPOINT_ONLY"
    ], var.adb_network_access_type)
    error_message = "adb_network_access_type must be PUBLIC_ENDPOINT, SECURE_ACCESS_FROM_ALLOWED_IPS_AND_VCNS, or PRIVATE_ENDPOINT_ONLY."
  }
}

variable "adb_use_private_subnet" {
  description = "Whether to use a private subnet for Autonomous Database"
  type        = bool
  default     = false
}

variable "adb_subnet_id" {
  description = "Private subnet OCID for Autonomous Database (used when private subnet is enabled)"
  type        = string
  default     = ""
}

variable "adb_acl_notation_type" {
  description = "Access-control notation for secure access from allowed IPs and VCNs"
  type        = string
  default     = "VCN"

  validation {
    condition     = contains(["VCN", "CIDR_BLOCK"], var.adb_acl_notation_type)
    error_message = "adb_acl_notation_type must be VCN or CIDR_BLOCK."
  }
}

variable "adb_acl_vcn_id" {
  description = "VCN OCID allowed to access Autonomous Database when secure ACL mode is selected"
  type        = string
  default     = ""
}

variable "adb_acl_subnet_id" {
  description = "Optional subnet OCID used to derive the CIDR entry for ADB VCN ACL mode"
  type        = string
  default     = ""
}

variable "adb_acl_cidr_blocks" {
  description = "Comma-separated CIDR blocks allowed to access Autonomous Database when CIDR ACL mode is selected"
  type        = string
  default     = ""
}

variable "adb_is_mtls_connection_required" {
  description = "Require mutual TLS (mTLS) connections for Autonomous Database"
  type        = bool
  default     = true
}

variable "instance_display_name" {
  default = "AIPOC_INSTANCE"
}

variable "application_port" {
  description = "TCP port exposed by the No.1-RAG application"
  type        = number
  default     = 8080

  validation {
    condition = (
      var.application_port == floor(var.application_port) &&
      var.application_port >= 1 &&
      var.application_port <= 65535 &&
      !contains([3000, 5432, 7932], var.application_port)
    )
    error_message = "application_port must be an integer between 1 and 65535, excluding reserved ports 3000, 5432, and 7932."
  }
}

variable "instance_shape" {
  default = "VM.Standard.E4.Flex"
}

variable "instance_flex_shape_ocpus" {
  type    = number
  default = 2

  validation {
    condition     = var.instance_flex_shape_ocpus > 0
    error_message = "instance_flex_shape_ocpus must be greater than 0."
  }
}

variable "instance_flex_shape_memory" {
  type    = number
  default = 16

  validation {
    condition     = var.instance_flex_shape_memory > 0
    error_message = "instance_flex_shape_memory must be greater than 0."
  }
}

variable "instance_boot_volume_size" {
  type    = number
  default = 100

  validation {
    condition     = var.instance_boot_volume_size >= 50 && var.instance_boot_volume_size <= 32768
    error_message = "instance_boot_volume_size must be between 50 and 32768 GB."
  }
}

variable "instance_boot_volume_vpus" {
  type    = number
  default = 10

  validation {
    condition     = contains(concat([10, 20], range(30, 121)), var.instance_boot_volume_vpus)
    error_message = "instance_boot_volume_vpus must be 10, 20, or a value from 30 through 120."
  }
}

variable "instance_image_source_id" {
  default = "ocid1.image.oc1.ap-tokyo-1.aaaaaaaaoiusqhftxmiyjlulnxx5mdnqfv6pjx4hdcoks3exn7gsrcwpkpdq"
}

variable "subnet_ai_subnet_id" {
  default = ""
}

variable "ssh_authorized_keys" {
  default = ""
}
