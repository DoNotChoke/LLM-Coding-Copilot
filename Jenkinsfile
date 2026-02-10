pipeline {
  agent any

  options {
    timestamps()
    disableConcurrentBuilds()
    // ansiColor('xterm')
  }

  parameters {
    booleanParam(name: 'FULL_INGEST', defaultValue: false, description: 'If true, ingest full repo scope (ignore changed files).')
    string(name: 'MILVUS_COLLECTION', defaultValue: 'code_chunks', description: 'Milvus collection name.')
    string(name: 'EMBED_MODEL', defaultValue: 'krlvi/sentence-t5-base-nlpl-code_search_net', description: 'Embedding model name/path.')
    string(name: 'EMBED_DIM', defaultValue: '768', description: 'Embedding dimension (must match Milvus schema).')
    string(name: 'BATCH_SIZE', defaultValue: '128', description: 'Batch size for embedding + Milvus insert.')
    string(name: 'COMPOSE_NETWORK', defaultValue: 'src_datastore', description: 'Docker network to reach Milvus (e.g. src_datastore).')

    // Case A: restrict ingest scope but keep repo_root="."
    string(name: 'INGEST_INCLUDE_DIRS', defaultValue: 'src', description: 'Comma-separated dirs (relative to repo root) to ingest. Example: src,tools,scripts')
  }

  environment {
    // pip cache persists because workspace is under /var/jenkins_home (volume)
    PIP_CACHE_DIR = "${WORKSPACE}/.pip-cache"

    REPO_ID = "${JOB_NAME}"

    // Milvus in docker-compose network
    MILVUS_HOST = "standalone"
    MILVUS_PORT = "19530"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
        sh 'git rev-parse HEAD'
      }
    }

    stage('Compute Changed Files') {
      when { expression { return !params.FULL_INGEST } }
      steps {
        script {
          def base = env.GIT_PREVIOUS_SUCCESSFUL_COMMIT
          if (!base || base.trim().isEmpty()) {
            base = sh(script: 'git rev-parse HEAD~1', returnStdout: true).trim()
          }
          def head = sh(script: 'git rev-parse HEAD', returnStdout: true).trim()

          def changed = sh(script: "git diff --name-only ${base} ${head} || true", returnStdout: true).trim()
          writeFile file: 'changed_files.txt', text: (changed ? changed + "\n" : "")
          env.CHANGED_FILES = changed
          echo "Changed files:\n${changed}"
        }
      }
    }

    stage('Ingest to Milvus (docker agent)') {
      agent {
        docker {
          image 'python:3.11-slim'
          reuseNode true
          args "--network ${params.COMPOSE_NETWORK}"
        }
      }
      steps {
        sh """
          set -euxo pipefail

          apt-get update
          apt-get install -y --no-install-recommends git ca-certificates
          rm -rf /var/lib/apt/lists/*

          python -m pip install -U pip
          pip install --cache-dir "${PIP_CACHE_DIR}" -r requirements.txt

          HEAD="\$(git rev-parse HEAD)"
          BRANCH="\${BRANCH_NAME:-\$(git rev-parse --abbrev-ref HEAD)}"

          if [ "${params.FULL_INGEST}" = "true" ]; then
            FULL_FLAG="--full"
            export CHANGED_FILES=""
          else
            FULL_FLAG=""
            export CHANGED_FILES="${env.CHANGED_FILES ?: ""}"
          fi

          python src/pipeline_ingest.py \\
            --repo_root . \\
            --include_dirs "${params.INGEST_INCLUDE_DIRS}" \\
            --repo "${env.REPO_ID}" \\
            --branch "\${BRANCH}" \\
            --commit "\${HEAD}" \\
            --milvus_host "${MILVUS_HOST}" \\
            --milvus_port "${MILVUS_PORT}" \\
            --collection "${params.MILVUS_COLLECTION}" \\
            --embed_model "${params.EMBED_MODEL}" \\
            --embed_dim "${params.EMBED_DIM}" \\
            --batch_size "${params.BATCH_SIZE}" \\
            \${FULL_FLAG}
        """
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'changed_files.txt', allowEmptyArchive: true, fingerprint: true
    }
  }
}
