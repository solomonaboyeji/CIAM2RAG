# atlas.hcl

# The database password. This is received as input from CLI,
# note that atlas can also read password from Secret Managers

variable "database_password" {
    type = string
}

# The database host
variable "database_host" {
    type = string
}

# The database port
variable "database_port" {
    type = number
}

# The database user' name
variable "database_username" {
    type = string
}

# The database's name
variable "database_name" {
    type = string
}

# The path to the SQLAlchemy models
variable "models_path" {
    type = string
}

# external datasource to get the schema from SQLAlchemy models
data "external_schema" "sqlalchemy" {
  program = [
    "atlas-provider-sqlalchemy",
    "--path", "${var.models_path}",
    "--dialect", "postgresql"
  ]
}

# configuration to run the migrations
env "run-migrations" {
  src = data.external_schema.sqlalchemy.url
  dev = "docker://postgres/15/dev?search_path=public" # a dev empty database spin up to setup the desired state we want
  migration {
    dir = "file://src/migrations"
  }
  format {
    migrate {
      diff = "{{ sql . \"  \" }}"
    }
  }
}

# the local environment (database) represents our local settings
env "local" {
  url = "postgres://${var.database_username}:${var.database_password}@${var.database_host}:${var.database_port}/${var.database_name}?search_path=public&sslmode=disable"
  migration {
    dir = "file://src/migrations"
    # dir = "atlas://sample-app" # uses the atlas.cloud directory
    # dir = "file://src/migrations" # uses the migrations directory
  }
}